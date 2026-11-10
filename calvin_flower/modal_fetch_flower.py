"""Download FLOWER CALVIN ABC checkpoint into the calvin-data Modal volume."""
import modal

app = modal.App("calvin-fetch-flower")
image = modal.Image.debian_slim(python_version="3.10").pip_install("huggingface_hub")
vol = modal.Volume.from_name("calvin-data", create_if_missing=True)


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def fetch():
    from huggingface_hub import snapshot_download
    p = snapshot_download("mbreuss/flower_calvin_abc", local_dir="/data/flower_calvin_abc")
    import os
    files = []
    for root, _, fs in os.walk(p):
        for f in fs:
            fp = os.path.join(root, f)
            files.append((fp.replace(p, ""), round(os.path.getsize(fp) / 1e6, 1)))
    vol.commit()
    return files


@app.local_entrypoint()
def main():
    for f, mb in fetch.remote():
        print(f, mb, "MB")
