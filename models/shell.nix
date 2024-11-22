with import <nixpkgs> {
  config.allowUnfree = true;
};
let
  pythonPackages = python312Packages;
  llvm = llvmPackages_latest;
in pkgs.mkShell{
  name = "barracuda";
  venvDir = "./.venv";

  buildInputs = [
    python312Full
    pythonPackages.pip
    pythonPackages.venvShellHook
    pythonPackages.numpy
    pythonPackages.pandas
    cudaPackages.cudatoolkit
    cudaPackages.cudnn

    taglib
    openssl
    libxml2
    libxslt
    libzip
    zlib

    cmake
    clang-tools
    llvm.libstdcxxClang
    llvm.libcxx
    ta-lib
  ];

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -r requirements.txt
  '';

  postShellHook = ''
    unset SOURCE_DATE_EPOCH
  '';
}

