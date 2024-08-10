use std::{
    env,
    fs::{create_dir_all, File},
    io::{Cursor, Read, Write},
    path::Path,
};
use zip::ZipArchive;

#[cfg(feature = "download-binaries")]
const ORT_EXTRACT_DIR: &str = "onnxruntime_extensions";

fn copy_file(src: &Path, dst: &Path) {
    if dst.exists() {
        std::fs::remove_file(dst).unwrap();
    }
    std::fs::copy(src, dst).unwrap();
}

fn get_cargo_target_dir() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    let profile = std::env::var("PROFILE")?;
    let mut target_dir = None;
    let mut sub_path = out_dir.as_path();
    while let Some(parent) = sub_path.parent() {
        if parent.ends_with(&profile) {
            target_dir = Some(parent);
            break;
        }
        sub_path = parent;
    }
    let target_dir = target_dir.ok_or("not found")?;
    Ok(target_dir.to_path_buf())
}

fn fetch_file(source_url: &str) -> Vec<u8> {
    let resp = ureq::AgentBuilder::new()
        .try_proxy_from_env(true)
        .build()
        .get(source_url)
        .timeout(std::time::Duration::from_secs(1800))
        .call()
        .unwrap_or_else(|err| panic!("Failed to GET `{source_url}`: {err}"));

    let len = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<usize>().ok())
        .expect("Content-Length header should be present on archive response");
    let mut reader = resp.into_reader();
    let mut buffer = Vec::new();
    reader
        .read_to_end(&mut buffer)
        .unwrap_or_else(|err| panic!("Failed to download from `{source_url}`: {err}"));
    assert_eq!(buffer.len(), len);
    buffer
}

pub fn extract_zip(data: &[u8], path: &Path) {
    let cursor = Cursor::new(data);

    let mut zipa = ZipArchive::new(cursor).unwrap();

    for i in 0..zipa.len() {
        let mut file = zipa.by_index(i).unwrap();

        if let Some(name) = file.enclosed_name() {
            let dest_path = path.join(name);
            if file.is_dir() {
                create_dir_all(&dest_path).unwrap();
                continue;
            }

            let parent = dest_path.parent().expect("Failed to get parent");

            if !parent.exists() {
                create_dir_all(parent).unwrap();
            }

            let mut buff: Vec<u8> = Vec::new();
            file.read_to_end(&mut buff).unwrap();
            let mut fileout = File::create(dest_path).expect("Failed to open file");

            fileout.write_all(&buff).unwrap();
        }
    }
}

fn main() {
    let target = env::var("TARGET").unwrap();
    let target_dir = get_cargo_target_dir().unwrap();
    let extract_path = target_dir.join(ORT_EXTRACT_DIR);
    if !extract_path.exists() {
        let compressed_path = fetch_file(
            "https://globalcdn.nuget.org/packages/microsoft.ml.onnxruntime.extensions.0.10.0.nupkg",
        );

        extract_zip(&compressed_path, &extract_path);
    }
    let lib_path = extract_path.join(if cfg!(windows) {
        "runtimes/win-x64/native/ortextensions.dll"
    } else if cfg!(target_os = "macos") {
        "runtimes/osx.10.14-arm64/native/libortextensions.dylib"
    } else {
        "runtimes/linux-x64/native/libortextensions.so"
    });
    let filename = lib_path.file_name().unwrap();
    println!("cargo:warning={}", lib_path.display());
    copy_file(&lib_path, &target_dir.join(filename));
    // Copy DLLs to examples as well
    if target_dir.join("examples").exists() {
        let dst = target_dir.join("examples").join(filename);
        if !dst.exists() {
            copy_file(&lib_path, &dst);
        }
    }
}
