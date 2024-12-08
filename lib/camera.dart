import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class CameraScreen extends StatefulWidget {
    @override
    _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
    late Future<List<CameraDescription>> _camerasFuture;
    late CameraController? _controller;
    late Future<void>? _initializeControllerFuture;

    @override
    void initState() {
        super.initState();
        _camerasFuture = availableCameras();
    }

    @override
    void dispose() {
        _controller?.dispose();
        super.dispose();
    }

    @override
    Widget build(BuildContext context) {
        return Scaffold(
            appBar: AppBar(title: const Text('Camera Screen')),
            body: FutureBuilder<List<CameraDescription>>(
                future: _camerasFuture,
                builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator()); // Show loading while initializing cameras
          } else if (snapshot.hasError) {
            // Log the error to the console for debugging
            print("Error initializing cameras: ${snapshot.error}");

            // Show a user-friendly error message
            return Center(
              child: Text(
                'Error: ${snapshot.error.toString()}',
                style: TextStyle(color: Colors.red),
              ),
            );
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
                  // Show the camera preview once the controller is initialized
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
                        // Ensure the camera is initialized
                        await _initializeControllerFuture;

                        // Capture the picture and save it to a file
                        final image = await _controller?.takePicture();

                        if (image != null) {
              // Display the captured image on a new screen
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => DisplayPictureScreen(imagePath: image.path),
                ),
              );
            }
                    } catch (e) {
                        print(e);
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

class PicturePreviewScreen extends StatelessWidget {
    final String imagePath;
    final Uint8List? imageBytes;

    const PicturePreviewScreen({super.key, required this.imagePath, this.imageBytes});

    @override
    Widget build(BuildContext context) {
        return Scaffold(
            appBar: AppBar(title: const Text('Picture Preview')),
            body: Column(
                children: [
                    Expanded(
                        child: kIsWeb
                        ? imageBytes != null
                        ? Image.memory(
                            imageBytes!,
                            fit: BoxFit.cover,
                        )
                        : const Center(child: Text('Image is not available'))
                        : Image.file(
                            File(imagePath),
                            fit: BoxFit.cover,
                        ),
                    ),
                    Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: ElevatedButton(
                            onPressed: () => Navigator.pop(context),
                            child: const Text('Retake Picture'),
                        ),
                    ),
                ],
            ),
        );
    }
}
