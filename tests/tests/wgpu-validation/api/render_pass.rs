//! Draw validation in render passes is cached across consecutive draws with
//! unchanged state. These tests check the cache is dropped by every command
//! that changes what a draw needs.

use wgpu_test::{fail, valid};

const SHADER: &str = "
    @group(0) @binding(0) var<uniform> u: vec4f;
    @vertex fn vs_buffers(@location(0) p: vec3f) -> @builtin(position) vec4f {
        return vec4f(p, 1.0) + u;
    }
    @vertex fn vs_plain() -> @builtin(position) vec4f { return u; }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
";

struct Fixture {
    device: wgpu::Device,
    /// Kept alive: the device only holds the queue weakly.
    _queue: wgpu::Queue,
    view: wgpu::TextureView,
    shader: wgpu::ShaderModule,
    layout: wgpu::PipelineLayout,
    /// Needs vertex buffer 0 and a uniform bind group at 0.
    pipeline: wgpu::RenderPipeline,
    /// Needs only the uniform bind group at 0.
    pipeline_no_vertex: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    /// Same binding slot, storage instead of uniform: incompatible with both pipelines.
    bind_group_other: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

fn setup() -> Fixture {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout_entry = |ty| wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[layout_entry(wgpu::BufferBindingType::Uniform)],
    });
    let bgl_other = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[layout_entry(wgpu::BufferBindingType::Storage {
            read_only: true,
        })],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let make_pipeline = |entry: &str, buffers: &[Option<wgpu::VertexBufferLayout>]| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
    };
    let pipeline = make_pipeline(
        "vs_buffers",
        &[Some(wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        })],
    );
    let pipeline_no_vertex = make_pipeline("vs_plain", &[]);
    let make_buffer = |usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage,
            mapped_at_creation: false,
        })
    };
    let uniform = make_buffer(wgpu::BufferUsages::UNIFORM);
    let storage = make_buffer(wgpu::BufferUsages::STORAGE);
    let make_bind_group = |layout, buffer: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    };
    Fixture {
        bind_group: make_bind_group(&bgl, &uniform),
        bind_group_other: make_bind_group(&bgl_other, &storage),
        vertex_buffer: make_buffer(wgpu::BufferUsages::VERTEX),
        index_buffer: make_buffer(wgpu::BufferUsages::INDEX),
        device,
        _queue: queue,
        view,
        shader,
        layout,
        pipeline,
        pipeline_no_vertex,
    }
}

impl Fixture {
    fn encode(&self, f: impl FnOnce(&mut wgpu::RenderPass<'_>)) -> wgpu::CommandEncoder {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::default()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            f(&mut pass);
        }
        encoder
    }
}

#[test]
fn consecutive_draws_with_unchanged_state_are_valid() {
    let f = setup();
    let encoder = f.encode(|pass| {
        pass.set_pipeline(&f.pipeline);
        pass.set_bind_group(0, &f.bind_group, &[]);
        pass.set_vertex_buffer(0, f.vertex_buffer.slice(..));
        pass.set_index_buffer(f.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        for _ in 0..4 {
            pass.draw(0..3, 0..1);
            pass.draw_indexed(0..3, 0, 0..1);
        }
        // Rebinding the same buffers is a no-op for the HAL encoder but must stay valid.
        pass.set_vertex_buffer(0, f.vertex_buffer.slice(..));
        pass.set_index_buffer(f.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..3, 0, 0..1);
    });
    valid(&f.device, || encoder.finish());
}

#[test]
fn bind_group_change_after_valid_draw_is_revalidated() {
    let f = setup();
    let encoder = f.encode(|pass| {
        pass.set_pipeline(&f.pipeline);
        pass.set_bind_group(0, &f.bind_group, &[]);
        pass.set_vertex_buffer(0, f.vertex_buffer.slice(..));
        pass.draw(0..3, 0..1);
        pass.set_bind_group(0, &f.bind_group_other, &[]);
        pass.draw(0..3, 0..1);
    });
    fail(
        &f.device,
        || encoder.finish(),
        Some("is not compatible with the corresponding"),
    );
}

#[test]
fn pipeline_change_after_valid_draw_is_revalidated() {
    let f = setup();
    let encoder = f.encode(|pass| {
        pass.set_pipeline(&f.pipeline_no_vertex);
        pass.set_bind_group(0, &f.bind_group, &[]);
        pass.draw(0..3, 0..1);
        pass.set_pipeline(&f.pipeline);
        pass.draw(0..3, 0..1);
    });
    fail(
        &f.device,
        || encoder.finish(),
        Some("requires vertex buffer 0 to be set"),
    );
}

#[test]
fn draw_families_are_validated_separately() {
    let f = setup();
    let encoder = f.encode(|pass| {
        pass.set_pipeline(&f.pipeline);
        pass.set_bind_group(0, &f.bind_group, &[]);
        pass.set_vertex_buffer(0, f.vertex_buffer.slice(..));
        pass.draw(0..3, 0..1);
        pass.draw_indexed(0..3, 0, 0..1);
    });
    fail(
        &f.device,
        || encoder.finish(),
        Some("Index buffer must be set"),
    );
}

#[test]
fn index_format_change_is_revalidated() {
    let f = setup();
    let pipeline_strip = f
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&f.layout),
            vertex: wgpu::VertexState {
                module: &f.shader,
                entry_point: Some("vs_plain"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &f.shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint16),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
    let encoder = f.encode(|pass| {
        pass.set_pipeline(&pipeline_strip);
        pass.set_bind_group(0, &f.bind_group, &[]);
        pass.set_index_buffer(f.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..3, 0, 0..1);
        pass.set_index_buffer(f.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..3, 0, 0..1);
    });
    fail(&f.device, || encoder.finish(), Some("strip"));
}
