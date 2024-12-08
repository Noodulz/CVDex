import 'dart:io';
import 'dart:typed_data';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class CameraScreen extends StatefulWidget {
  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late Future<List<CameraDescription>> _camerasFuture;
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;

  @override
  void initState() {
    super.initState();
    _requestPermissions();
  }

  // Request camera permissions
  void _requestPermissions() async {
    PermissionStatus status = await Permission.camera.request();

    if (status.isGranted) {
      _camerasFuture = availableCameras();
      setState(() {});
    } else {
      // Handle the case where permission is denied
      print("Camera permission denied");
    }
  }

  @override
  void dispose() {
    _controller?.dispose(); // Dispose of the controller when the widget is disposed
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Camera Screen')),
      body: FutureBuilder<List<CameraDescription>>(
        future: _camerasFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            // Log the error
            print("Error initializing cameras: ${snapshot.error}");
            return Center(child: Text('Error: ${snapshot.error}'));
          } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
            return Center(child: Text('No cameras found'));
          } else {
            // Initialize the CameraController with the first available camera
            if (_controller == null) {
              _controller = CameraController(
                snapshot.data![0], // Use the first camera
                ResolutionPreset.high,
              );
              _initializeControllerFuture = _controller?.initialize();
            }

            return FutureBuilder<void>(
              future: _initializeControllerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  return CameraPreview(_controller!);
                } else {
                  return Center(child: CircularProgressIndicator());
                }
              },
            );
          }
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          try {
            if (_controller == null || !_controller!.value.isInitialized) {
              print('Camera not initialized');
              return;
            }

            await _initializeControllerFuture;

            final image = await _controller?.takePicture();
            if (image != null) {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => DisplayPictureScreen(imagePath: image.path),
                ),
              );
            }
          } catch (e) {
            print('Error capturing picture: $e');
          }
        },
        child: Icon(Icons.camera_alt),
      ),
    );
  }
}


class DisplayPictureScreen extends StatelessWidget {
  final String imagePath;

  DisplayPictureScreen({required this.imagePath});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Captured Image')),
      body: Center(
        child: Image.file(File(imagePath)),
      ),
    );
  }
}
