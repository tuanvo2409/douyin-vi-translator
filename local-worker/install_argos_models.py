"""Download local Argos Translate models used by the worker.

Run this once after `pip install -r requirements.txt`. Argos can resolve a
Chinese → Vietnamese path through installed intermediate models when a direct
pair is unavailable.
"""
import argostranslate.package


def install_pair(source: str, target: str) -> None:
    packages = argostranslate.package.get_available_packages()
    package = next((item for item in packages if item.from_code == source and item.to_code == target), None)
    if package is None:
        print(f"Không tìm thấy model Argos {source} → {target}; bỏ qua.")
        return
    path = package.download()
    argostranslate.package.install_from_path(path)
    print(f"Đã cài Argos model {source} → {target}")


if __name__ == "__main__":
    argostranslate.package.update_package_index()
    install_pair("zh", "vi")
    # Fallback path if the direct pair is unavailable in the current Argos index.
    install_pair("zh", "en")
    install_pair("en", "vi")
