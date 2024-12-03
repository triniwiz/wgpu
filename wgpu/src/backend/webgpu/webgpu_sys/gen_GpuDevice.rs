// DO NOT EDIT THIS FILE!
//
// This module part of a subset of web-sys that is used by wgpu's webgpu backend.
//
// These bindings are vendored into wgpu for the sole purpose of letting
// us pin the WebGPU backend to a specific version of the bindings, not
// to enable local changes. There are no provisions to preserve changes
// you make here the next time we re-vendor the bindings.
//
// The `web-sys` crate does not treat breaking changes to the WebGPU API
// as semver breaking changes, as WebGPU is "unstable". This means Cargo
// will not let us mix versions of `web-sys`, pinning WebGPU bindings to
// a specific version, while letting other bindings like WebGL get
// updated. Vendoring WebGPU was the workaround we chose.
//
// Vendoring also allows us to avoid building `web-sys` with
// `--cfg=web_sys_unstable_apis`, needed to get the WebGPU bindings.
//
// If you want to improve the generated code, please submit a PR to the https://github.com/rustwasm/wasm-bindgen repository.
//
// This file was generated by the `cargo xtask vendor-web-sys --version 0.2.97` command.
#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = EventTarget , extends = :: js_sys :: Object , js_name = GPUDevice , typescript_type = "GPUDevice")]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `GpuDevice` class."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type GpuDevice;

    # [wasm_bindgen (structural , method , getter , js_class = "GPUDevice" , js_name = features)]
    #[doc = "Getter for the `features` field of this object."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/features)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuSupportedFeatures`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn features(this: &GpuDevice) -> GpuSupportedFeatures;

    # [wasm_bindgen (structural , method , getter , js_class = "GPUDevice" , js_name = limits)]
    #[doc = "Getter for the `limits` field of this object."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/limits)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuSupportedLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn limits(this: &GpuDevice) -> GpuSupportedLimits;

    # [wasm_bindgen (structural , method , getter , js_class = "GPUDevice" , js_name = adapterInfo)]
    #[doc = "Getter for the `adapterInfo` field of this object."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/adapterInfo)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuAdapterInfo`, `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn adapter_info(this: &GpuDevice) -> GpuAdapterInfo;

    # [wasm_bindgen (structural , method , getter , js_class = "GPUDevice" , js_name = queue)]
    #[doc = "Getter for the `queue` field of this object."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/queue)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuQueue`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn queue(this: &GpuDevice) -> GpuQueue;

    # [wasm_bindgen (structural , method , getter , js_class = "GPUDevice" , js_name = lost)]
    #[doc = "Getter for the `lost` field of this object."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/lost)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn lost(this: &GpuDevice) -> ::js_sys::Promise;

    # [wasm_bindgen (structural , method , getter , js_class = "GPUDevice" , js_name = onuncapturederror)]
    #[doc = "Getter for the `onuncapturederror` field of this object."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/onuncapturederror)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn onuncapturederror(this: &GpuDevice) -> Option<::js_sys::Function>;

    # [wasm_bindgen (structural , method , setter , js_class = "GPUDevice" , js_name = onuncapturederror)]
    #[doc = "Setter for the `onuncapturederror` field of this object."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/onuncapturederror)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn set_onuncapturederror(this: &GpuDevice, value: Option<&::js_sys::Function>);

    # [wasm_bindgen (structural , method , getter , js_class = "GPUDevice" , js_name = label)]
    #[doc = "Getter for the `label` field of this object."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/label)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn label(this: &GpuDevice) -> ::alloc::string::String;

    # [wasm_bindgen (structural , method , setter , js_class = "GPUDevice" , js_name = label)]
    #[doc = "Setter for the `label` field of this object."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/label)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn set_label(this: &GpuDevice, value: &str);

    # [wasm_bindgen (method , structural , js_class = "GPUDevice" , js_name = createBindGroup)]
    #[doc = "The `createBindGroup()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createBindGroup)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuBindGroup`, `GpuBindGroupDescriptor`, `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_bind_group(this: &GpuDevice, descriptor: &GpuBindGroupDescriptor)
        -> GpuBindGroup;

    # [wasm_bindgen (catch , method , structural , js_class = "GPUDevice" , js_name = createBindGroupLayout)]
    #[doc = "The `createBindGroupLayout()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createBindGroupLayout)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuBindGroupLayout`, `GpuBindGroupLayoutDescriptor`, `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_bind_group_layout(
        this: &GpuDevice,
        descriptor: &GpuBindGroupLayoutDescriptor,
    ) -> Result<GpuBindGroupLayout, JsValue>;

    # [wasm_bindgen (catch , method , structural , js_class = "GPUDevice" , js_name = createBuffer)]
    #[doc = "The `createBuffer()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createBuffer)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuBuffer`, `GpuBufferDescriptor`, `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_buffer(
        this: &GpuDevice,
        descriptor: &GpuBufferDescriptor,
    ) -> Result<GpuBuffer, JsValue>;

    # [wasm_bindgen (method , structural , js_class = "GPUDevice" , js_name = createCommandEncoder)]
    #[doc = "The `createCommandEncoder()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createCommandEncoder)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuCommandEncoder`, `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_command_encoder(this: &GpuDevice) -> GpuCommandEncoder;

    # [wasm_bindgen (method , structural , js_class = "GPUDevice" , js_name = createCommandEncoder)]
    #[doc = "The `createCommandEncoder()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createCommandEncoder)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuCommandEncoder`, `GpuCommandEncoderDescriptor`, `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_command_encoder_with_descriptor(
        this: &GpuDevice,
        descriptor: &GpuCommandEncoderDescriptor,
    ) -> GpuCommandEncoder;

    # [wasm_bindgen (method , structural , js_class = "GPUDevice" , js_name = createComputePipeline)]
    #[doc = "The `createComputePipeline()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createComputePipeline)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuComputePipeline`, `GpuComputePipelineDescriptor`, `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_compute_pipeline(
        this: &GpuDevice,
        descriptor: &GpuComputePipelineDescriptor,
    ) -> GpuComputePipeline;

    # [wasm_bindgen (method , structural , js_class = "GPUDevice" , js_name = createComputePipelineAsync)]
    #[doc = "The `createComputePipelineAsync()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createComputePipelineAsync)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuComputePipelineDescriptor`, `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_compute_pipeline_async(
        this: &GpuDevice,
        descriptor: &GpuComputePipelineDescriptor,
    ) -> ::js_sys::Promise;

    # [wasm_bindgen (method , structural , js_class = "GPUDevice" , js_name = createPipelineLayout)]
    #[doc = "The `createPipelineLayout()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createPipelineLayout)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuPipelineLayout`, `GpuPipelineLayoutDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_pipeline_layout(
        this: &GpuDevice,
        descriptor: &GpuPipelineLayoutDescriptor,
    ) -> GpuPipelineLayout;

    # [wasm_bindgen (catch , method , structural , js_class = "GPUDevice" , js_name = createQuerySet)]
    #[doc = "The `createQuerySet()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createQuerySet)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuQuerySet`, `GpuQuerySetDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_query_set(
        this: &GpuDevice,
        descriptor: &GpuQuerySetDescriptor,
    ) -> Result<GpuQuerySet, JsValue>;

    # [wasm_bindgen (catch , method , structural , js_class = "GPUDevice" , js_name = createRenderBundleEncoder)]
    #[doc = "The `createRenderBundleEncoder()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createRenderBundleEncoder)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuRenderBundleEncoder`, `GpuRenderBundleEncoderDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_render_bundle_encoder(
        this: &GpuDevice,
        descriptor: &GpuRenderBundleEncoderDescriptor,
    ) -> Result<GpuRenderBundleEncoder, JsValue>;

    # [wasm_bindgen (catch , method , structural , js_class = "GPUDevice" , js_name = createRenderPipeline)]
    #[doc = "The `createRenderPipeline()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createRenderPipeline)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuRenderPipeline`, `GpuRenderPipelineDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_render_pipeline(
        this: &GpuDevice,
        descriptor: &GpuRenderPipelineDescriptor,
    ) -> Result<GpuRenderPipeline, JsValue>;

    # [wasm_bindgen (method , structural , js_class = "GPUDevice" , js_name = createRenderPipelineAsync)]
    #[doc = "The `createRenderPipelineAsync()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createRenderPipelineAsync)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuRenderPipelineDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_render_pipeline_async(
        this: &GpuDevice,
        descriptor: &GpuRenderPipelineDescriptor,
    ) -> ::js_sys::Promise;

    # [wasm_bindgen (method , structural , js_class = "GPUDevice" , js_name = createSampler)]
    #[doc = "The `createSampler()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createSampler)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuSampler`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_sampler(this: &GpuDevice) -> GpuSampler;

    # [wasm_bindgen (method , structural , js_class = "GPUDevice" , js_name = createSampler)]
    #[doc = "The `createSampler()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createSampler)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuSampler`, `GpuSamplerDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_sampler_with_descriptor(
        this: &GpuDevice,
        descriptor: &GpuSamplerDescriptor,
    ) -> GpuSampler;

    # [wasm_bindgen (method , structural , js_class = "GPUDevice" , js_name = createShaderModule)]
    #[doc = "The `createShaderModule()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createShaderModule)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuShaderModule`, `GpuShaderModuleDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_shader_module(
        this: &GpuDevice,
        descriptor: &GpuShaderModuleDescriptor,
    ) -> GpuShaderModule;

    # [wasm_bindgen (catch , method , structural , js_class = "GPUDevice" , js_name = createTexture)]
    #[doc = "The `createTexture()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createTexture)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuTexture`, `GpuTextureDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_texture(
        this: &GpuDevice,
        descriptor: &GpuTextureDescriptor,
    ) -> Result<GpuTexture, JsValue>;

    # [wasm_bindgen (method , structural , js_class = "GPUDevice" , js_name = destroy)]
    #[doc = "The `destroy()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/destroy)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn destroy(this: &GpuDevice);

    # [wasm_bindgen (catch , method , structural , js_class = "GPUDevice" , js_name = importExternalTexture)]
    #[doc = "The `importExternalTexture()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/importExternalTexture)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuExternalTexture`, `GpuExternalTextureDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn import_external_texture(
        this: &GpuDevice,
        descriptor: &GpuExternalTextureDescriptor,
    ) -> Result<GpuExternalTexture, JsValue>;

    # [wasm_bindgen (method , structural , js_class = "GPUDevice" , js_name = popErrorScope)]
    #[doc = "The `popErrorScope()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/popErrorScope)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn pop_error_scope(this: &GpuDevice) -> ::js_sys::Promise;

    # [wasm_bindgen (method , structural , js_class = "GPUDevice" , js_name = pushErrorScope)]
    #[doc = "The `pushErrorScope()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/pushErrorScope)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `GpuErrorFilter`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn push_error_scope(this: &GpuDevice, filter: GpuErrorFilter);
}
