<h1 align="center" id="home">CVDex</h1>

<div align="center" id="badges">

  <img href="https://google.com" src="https://img.shields.io/badge/Flutter-%2302569B.svg?style=for-the-badge&logo=Flutter&logoColor=white" alt="Flutter">

  <img href="https://dart.dev" src="https://img.shields.io/badge/dart-%230175C2.svg?style=for-the-badge&logo=dart&logoColor=white" alt="Dart">

  <img href="https://www.pytorch.org/" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  
  <img href="https://firebase.google.com/" src="https://img.shields.io/badge/firebase-a08021?style=for-the-badge&logo=firebase&logoColor=ffcd34" alt="Firebase">
  
  <img href="https://www.android.com/" src="https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white" alt="Android">

  <img href="https://developer.android.com/studio" src="https://img.shields.io/badge/android%20studio-346ac1?style=for-the-badge&logo=android%20studio&logoColor=white" alt="Android Studio">
</div>
<br>

> **CVDEX is a Flutter-based Pokédex app that lets users scan Pokémon images in real life, leveraging a PyTorch model, Firebase integration, and a clean UI inspired by the original Pokédex. Users can build their digital Pokédex while enjoying a gamified experience.**

<div align="center" id="demo">
  <img src="https://github.com/Noodulz/CVDex/blob/main/assets/pidgey-demo.gif" alt="Pidgey" width=232 height=480 style="margin-right: 20px;">
  <img src="https://github.com/Noodulz/CVDex/blob/main/assets/vaporeon-demo.gif" alt="Vaporeon" style="margin-left:20px;">
</div>

# How to Run

## Prerequisites
Android SDK (>= 34), Android Studio (>= Ladybug version 2024.2), Android emulators, Python 3 (>= 3.6), PyTorch 2.x, Gradle, latest Flutter version (>= 3.24.5), Java JDK 17 or higher (21 recommended)

## Starting the Server
To run the PyTorch model server, navigate to `models/`, install all dependencies with `pip install -r requirements.txt`, and then run `python3 src/server.py` which loads the model from Kaggle on a Flask server ready for handling requests. 

## Building for Android
Before every build and testing every change, run `flutter clean` before `flutter pub get` and then `flutter build apk`. After the APK is built, launch an emulator of your choice by checking `flutter emulators` and then `flutter emulators --launch <emulator id>`. After the emulator is launched, install the APK into the emulator with `adb install build/app/outputs/flutter-apk/app-release.apk`. Then once that's done the app can be launched automatically with `adb shell monkey -p com.example.cvdex 1`.

## Building for Web
When cloning, be sure to run `flutter pub get` before `flutter run` with any additional parameters. For chrome the parameters should be `-d chrome` after `flutter run`. Note that for Chrome as far as we know, camera integration is not yet enabled so photo uploads will happen by prompting the file explorer instead. To compile a release, run `flutter build web --release`, navigate to `build/web` and spin up a web server with `python3 -m http.server 8000` (npm http-server also works). 
