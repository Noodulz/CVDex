with import <nixpkgs> {
  config.allowUnfree = true;
};
let
  pythonPackages = python311Packages;
  llvm = llvmPackages_latest;
in pkgs.mkShell{
  name = "barracuda";
  venvDir = "./.venv";

  buildInputs = [
    python311Full
    pythonPackages.pip
    pythonPackages.venvShellHook
    pythonPackages.numpy
    pythonPackages.pandas
    pythonPackages.pytorch-bin
    pythonPackages.torchvision-bin
    pythonPackages.torchaudio-bin
    pythonPackages.kaggle
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

  shellHook = ''
    echo "Setting up CUDA environment"
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
  '';

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -r requirements.txt
  '';

  postShellHook = ''
    unset SOURCE_DATE_EPOCH
  '';
}

