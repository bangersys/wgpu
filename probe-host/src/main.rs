//! Throwaway host-consumption probe for the Row-1 borrow accessor.
//!
//! NOT shippable code. It answers whether the borrowed GL name denotes a
//! texture the host can actually consume: the host owns the GL context and
//! the texture, writes a pattern with raw GL, imports the name into
//! wgpu-hal, borrows it back through `native_texture_handle`, and reads the
//! pixels back with raw GL. Same thread throughout; no submit, no fences,
//! so dead fence drivers cannot block it.
//!
//! Needs a real GL driver. Windows default is WGL native GL; rebuild with
//! `RUSTFLAGS="--cfg windows_angle"` for the ANGLE configuration.
//!
//! ```sh
//! cd probe-host
//! cargo run
//! ```
//!
//! Paste the full output back. Exit 0 plus PASS closes the question.

use std::ffi::CString;
use std::num::NonZeroU32;

use glow::HasContext;
use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
use glutin::display::{Display, DisplayApiPreference};
use glutin::prelude::*;
use glutin::surface::{PbufferSurface, SurfaceAttributesBuilder};
use raw_window_handle::{RawDisplayHandle, WindowsDisplayHandle};
use wgpu_hal::{Adapter as _, Device as _};

fn main() {
    // Host-owned display: WGL-first so the configuration matches wgpu's own
    // default backend selection on Windows.
    let raw_display = RawDisplayHandle::Windows(WindowsDisplayHandle::new());
    let display = unsafe {
        Display::new(raw_display, DisplayApiPreference::Wgl(None))
            .expect("FAIL: no GL display (no driver)")
    };
    // OpenGL configs without a surface-type filter: filtering to PBUFFER up
    // front leaves a single exotic config on picky WGL drivers. Try each
    // config in turn and keep the first one that accepts BOTH context
    // creation and pbuffer surface creation.
    let template = ConfigTemplateBuilder::new()
        .with_api(glutin::config::Api::OPENGL)
        .build();
    let context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
        .build(None);
    let surface_attributes = SurfaceAttributesBuilder::<PbufferSurface>::new()
        .build(NonZeroU32::new(64).unwrap(), NonZeroU32::new(64).unwrap());
    let mut tried_configs = 0;
    let mut tried_surfaces = 0;
    let mut working = None;
    for config in
        unsafe { display.find_configs(template) }.expect("FAIL: GL config enumeration failed")
    {
        tried_configs += 1;
        let context = match unsafe { display.create_context(&config, &context_attributes) } {
            Ok(context) => context,
            Err(_) => continue,
        };
        tried_surfaces += 1;
        match unsafe { display.create_pbuffer_surface(&config, &surface_attributes) } {
            Ok(surface) => {
                working = Some((config, context, surface));
                break;
            }
            Err(_) => continue,
        }
    }
    let (config, context, surface) = working.unwrap_or_else(|| {
        println!("FAIL: no usable GL config (tried {tried_configs} configs, {tried_surfaces} pbuffer surfaces)");
        std::process::exit(6);
    });
    println!(
        "gl config accepted (configs tried: {tried_configs}, surfaces tried: {tried_surfaces})"
    );
    let context = context
        .make_current(&surface)
        .expect("FAIL: cannot make GL context current");

    // Host-owned GL function table on the host-owned current context.
    let gl = unsafe {
        glow::Context::from_loader_function(|s| {
            let name = CString::new(s).unwrap();
            display.get_proc_address(&name) as *const _
        })
    };
    println!(
        "host GL: {} / {} / {}",
        unsafe { gl.get_parameter_string(glow::VERSION) },
        unsafe { gl.get_parameter_string(glow::VENDOR) },
        unsafe { gl.get_parameter_string(glow::RENDERER) },
    );
    unsafe {
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
    }

    // Host-owned texture with a known pattern, written with raw GL only.
    const W: i32 = 64;
    const H: i32 = 64;
    let mut pattern = vec![0u8; (W * H * 4) as usize];
    for (i, b) in pattern.iter_mut().enumerate() {
        *b = (i.wrapping_mul(5).wrapping_add(11)) as u8;
    }
    let host_name = unsafe {
        let name = gl.create_texture().expect("FAIL: host gen_texture failed");
        gl.bind_texture(glow::TEXTURE_2D, Some(name));
        gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::RGBA8, W, H);
        gl.tex_sub_image_2d(
            glow::TEXTURE_2D,
            0,
            0,
            0,
            W,
            H,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(&pattern)),
        );
        gl.bind_texture(glow::TEXTURE_2D, None);
        let id: NonZeroU32 = name.0;
        println!("host texture name: {id}");
        id
    };

    // Lend the host context to wgpu-hal. The adapter borrows; the host keeps
    // owning context and texture for the whole run.
    let exposed = unsafe {
        <wgpu_hal::api::Gles as wgpu_hal::Api>::Adapter::new_external(
            |name| {
                let cname = CString::new(name).unwrap();
                display.get_proc_address(&cname) as *const _
            },
            wgpu_types::GlBackendOptions::default(),
        )
    }
    .expect("FAIL: hal adapter rejected the host context");
    let opened = unsafe {
        exposed.adapter.open(
            wgpu_types::Features::empty(),
            &wgpu_types::Limits::downlevel_defaults(),
            &wgpu_types::MemoryHints::default(),
        )
    }
    .expect("FAIL: hal device open failed");

    // Import the host name with a drop callback: wgpu must never delete a
    // host-owned texture, so the callback proves the ownership direction.
    let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let dropped_flag = dropped.clone();
    let hal_texture = unsafe {
        opened.device.texture_from_raw(
            host_name,
            &wgpu_hal::TextureDescriptor {
                label: None,
                size: wgpu_types::Extent3d {
                    width: W as u32,
                    height: H as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu_types::TextureDimension::D2,
                format: wgpu_types::TextureFormat::Rgba8Unorm,
                usage: wgpu_types::TextureUses::RESOURCE,
                memory_flags: wgpu_hal::MemoryFlags::empty(),
                view_formats: vec![],
            },
            Some(Box::new(move || {
                dropped_flag.store(true, std::sync::atomic::Ordering::SeqCst);
            })),
        )
    };

    // The question: does the borrow return the host's own name?
    let borrowed = opened.device.native_texture_handle(&hal_texture);
    println!("hal borrow: {borrowed:?}");
    assert_eq!(
        borrowed.map(|t| t.0),
        Some(host_name),
        "FAIL: borrow did not return the host texture name"
    );

    // Host-side consumption on the still-current host context: finish the
    // (same-thread) GL stream, then read the pixels through the borrowed
    // name and compare against the written pattern.
    let mut readback = vec![0u8; pattern.len()];
    unsafe {
        gl.finish();
        gl.bind_texture(glow::TEXTURE_2D, Some(glow::NativeTexture(host_name)));
        gl.get_tex_image(
            glow::TEXTURE_2D,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelPackData::Slice(Some(&mut readback)),
        );
        gl.bind_texture(glow::TEXTURE_2D, None);
    }
    assert_eq!(readback.len(), pattern.len(), "FAIL: readback size differs");
    match readback
        .iter()
        .zip(pattern.iter())
        .position(|(a, b)| a != b)
    {
        None => println!("host readback: {} bytes identical", readback.len()),
        Some(i) => {
            println!(
                "FAIL: pixel mismatch at byte {i}: got {}, want {}",
                readback[i], pattern[i]
            );
            std::process::exit(4);
        }
    }

    // Ownership direction: dropping the hal texture must fire the callback
    // (wgpu releases) while the GL object survives for the host to delete.
    unsafe { opened.device.destroy_texture(hal_texture) };
    assert!(
        dropped.load(std::sync::atomic::Ordering::SeqCst),
        "FAIL: drop callback did not fire on hal texture destroy"
    );
    println!("drop callback: fired, host texture not deleted by wgpu");
    unsafe {
        gl.delete_texture(glow::NativeTexture(host_name));
    }

    println!("PASS: host wrote, hal borrowed, host read back identical");
    let _ = context;
}
