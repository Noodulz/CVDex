import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'app.dart';
import 'firebase_options.dart';
import 'package:image_picker_platform_interface/image_picker_platform_interface.dart';

class MyCameraDelegate extends ImagePickerCameraDelegate {
    @override
    Future<XFile?> takePhoto(
    {ImagePickerCameraDelegateOptions options =
        const ImagePickerCameraDelegateOptions()}) async {
        return _takeAPhoto(options.preferredCameraDevice);
    }
    @override
    Future<XFile?> takeVideo(
    {ImagePickerCameraDelegateOptions options =
        const ImagePickerCameraDelegateOptions()}) async {
        return _takeAVideo(options.preferredCameraDevice);
    }

    Future<XFile?> _takeAPhoto(CameraDevice device) async {
        print("Taking a photo");
        try{
            final cameras = await availableCameras();
            final CameraController controller = CameraController(
                cameras.first,
                ResolutionPreset.high,
            );

            await controller.initialize();
            final XFile photo = await controller.takePicture();

            controller.dispose();
            return photo;

        } catch (e){
            print("Error in getting cameras");
            return null;
        }
        return null; // Placeholder, replace with actual XFile object
    }

    Future<XFile?> _takeAVideo(CameraDevice device) async {
        // Implement your custom video capture logic here
        return null; // Placeholder, replace with actual XFile object
    }
}

void setUpCameraDelegate() {
    print("Camera delegate");
    final ImagePickerPlatform instance = ImagePickerPlatform.instance;
    if (instance is CameraDelegatingImagePickerPlatform) {
    instance.cameraDelegate = MyCameraDelegate();
    print("Custom camera delegate set up correctly");
  }
}

void main() async {
    WidgetsFlutterBinding.ensureInitialized();

    await Firebase.initializeApp(
        options: DefaultFirebaseOptions.currentPlatform,
    );

    setUpCameraDelegate();
    runApp(const MyApp());
}
