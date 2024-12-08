import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:mime/mime.dart';
import 'dart:io';
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: CameraPage(),
    );
  }
}

class CameraPage extends StatefulWidget {
  @override
  _CameraPageState createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  final ImagePicker _picker = ImagePicker();
  File? _image;

  Future<void> _takePicture() async {
    try {
      final XFile? photo = await _picker.pickImage(source: ImageSource.camera);
      if (photo != null) {
        setState(() {
          _image = File(photo.path);
        });

        await _imageToServer(photo);
      }
    } catch (e) {
      print('Error: $e');
    }
  }

  Future<void> _imageToServer(XFile image) async {
    try {
      final bytes = await image.readAsBytes();
      final mimeType = lookupMimeType(image.path) ?? 'application/octet-stream';
      // Prepare the multipart request
      final uri = Uri.parse('http://localhost:5000/predict');
      final request = http.MultipartRequest('POST', uri);

      // Add the image as a file to the multipart request
      request.files.add(http.MultipartFile.fromBytes(
        'image', // The field name expected by the server
        bytes,
        filename: image.name,
        contentType: MediaType.parse(mimeType),
      ));

      // Send the request
      final response = await request.send();

      // Handle the server response
      if (response.statusCode == 200) {
        print('Image successfully uploaded!');
        final responseData = await response.stream.bytesToString();
        print('Server Response: $responseData');
      } else {
        print('Failed to upload image: ${response.statusCode}');
      }
    } catch (e) {
      print('Error sending image to server: $e');
    }
  }

  Future<void> _pickImageFromGallery() async {
    try {
      final XFile? photo = await _picker.pickImage(source: ImageSource.gallery);
      if (photo != null) {
        setState(() {
          _image = File(photo.path);
        });
      }
    } catch (e) {
      print('Error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Camera Page'),
      ),
      body: Stack(
        children: [
          // Center the image or placeholder text
          Center(
            child: _image == null
                ? Text(
                    'Upload from your gallery or snap a photo!',
                    style: TextStyle(fontSize: 18, color: Colors.grey),
                  )
                : Image.network(_image!.path),
          ),

          // Position buttons at the bottom center
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding:
                  const EdgeInsets.only(bottom: 20.0), // Adjust bottom margin
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // Gallery Button
                  FloatingActionButton(
                    onPressed: _pickImageFromGallery,
                    heroTag: 'galleryButton',
                    child: Icon(Icons.photo_library),
                  ),
                  SizedBox(width: 20), // Spacing between buttons

                  // Camera Button
                  FloatingActionButton(
                    onPressed: _takePicture,
                    heroTag: 'cameraButton',
                    child: Icon(Icons.camera),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
