//! Many small `Queue::write_buffer` calls: they share staging chunks, which
//! fill up mid-batch and are reused across submissions.

use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(QUEUE_WRITE_POOLED);
}

const SIZE: u64 = 3 << 20;

fn pattern(round: u32, offset: u64, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| {
            ((offset as usize + i) as u32)
                .wrapping_mul(31)
                .wrapping_add(round * 7) as u8
        })
        .collect()
}

async fn read_back(ctx: &TestingContext, src: &wgpu::Buffer) -> Vec<u8> {
    let cpu = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cpu"),
        size: SIZE,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(src, 0, &cpu, 0, SIZE);
    ctx.queue.submit(Some(encoder.finish()));
    cpu.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    let data = cpu.slice(..).get_mapped_range().unwrap().to_vec();
    data
}

#[apply(gpu_test!)]
static QUEUE_WRITE_POOLED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| async move {
        let gpu = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu"),
            size: SIZE,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Mostly small writes, plus some past the pooled limit.
        let sizes = [4u64, 64, 256, 4096, 70_000, 300_000];
        for round in 0..8u32 {
            let mut expected = vec![0u8; SIZE as usize];
            if round > 0 {
                expected = read_back(&ctx, &gpu).await;
            }
            let mut offset = 0u64;
            let mut i = 0usize;
            while offset < SIZE {
                let len = sizes[(i + round as usize) % sizes.len()].min(SIZE - offset);
                let data = pattern(round, offset, len as usize);
                ctx.queue.write_buffer(&gpu, offset, &data);
                expected[offset as usize..(offset + len) as usize].copy_from_slice(&data);
                offset += len;
                i += 1;
            }
            // A later write in the same batch wins.
            let late = vec![0xA5u8; 1024];
            ctx.queue.write_buffer(&gpu, 4096, &late);
            expected[4096..4096 + 1024].copy_from_slice(&late);

            if round % 2 == 0 {
                // Leave chunks in flight: the next round reuses or grows the pool.
                ctx.queue.submit(None);
            }
            let actual = read_back(&ctx, &gpu).await;
            if let Some(at) = actual.iter().zip(&expected).position(|(a, e)| a != e) {
                panic!(
                    "round {round}: byte {at} is {} but should be {}",
                    actual[at], expected[at]
                );
            }
        }
    });
