<h1 align="center" id="home">CVDex</h1>

<div align="center" id="badges">
  <a href="https://flutter.dev" target="_blank">
    <img src="https://img.shields.io/badge/Flutter-%2302569B.svg?style=for-the-badge&logo=Flutter&logoColor=white" alt="Flutter">
  </a>
  <a href="https://dart.dev" target="_blank">
    <img src="https://img.shields.io/badge/dart-%230175C2.svg?style=for-the-badge&logo=dart&logoColor=white" alt="Dart">
  </a>
  <a href="https://www.tensorflow.org/" target="_blank">
    <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" alt="TensorFlow">
  </a>
  <a href="https://firebase.google.com/" target="_blank">
    <img src="https://img.shields.io/badge/firebase-a08021?style=for-the-badge&logo=firebase&logoColor=ffcd34" alt="Firebase">
  </a>
  <a href="https://www.android.com/" target="_blank">
    <img src="https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white" alt="Android">
  </a>
  <a href="https://developer.android.com/studio" target="_blank">
    <img src="https://img.shields.io/badge/android%20studio-346ac1?style=for-the-badge&logo=android%20studio&logoColor=white" alt="Android Studio">
  </a>
  <a href="https://developer.apple.com/ios/" target="_blank">
    <img src="https://img.shields.io/badge/iOS-000000?style=for-the-badge&logo=ios&logoColor=white" alt="iOS">
  </a>
</div>


A Pokédex clone made in Flutter and Dart where users can scan Pokémon images in real life and store their entries. Like a real Pokédex!

# How to Run

## Prerequisites
Android Studio (>= Ladybug version), Android emulators, Gradle, latest Flutter version, Java JDK 17 or higher (21 recommended)

## Building for Android
Before every build and testing every change, run `flutter clean` before `flutter pub get` and then `flutter build apk`. After the APK is built, launch an emulator of your choice by checking `flutter emulators` and then `flutter emulators --launch <emulator id>`. After the emulator is launched, install the APK into the emulator with `adb install build/app/outputs/flutter-apk/app-release.apk`. Then once that's done the app can be launched automatically with `adb shell monkey -p com.example.cvdex 1`.

## Building for Web
When cloning, be sure to run `flutter pub get` before `flutter run` with any additional parameters. For chrome the parameters should be `-d chrome` after `flutter run`. Note that for Chrome as far as we know, camera integration is not yet enabled so photo uploads will happen by prompting the file explorer instead. 
