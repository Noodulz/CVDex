import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:mime/mime.dart';
import 'dart:io';
import 'dart:convert';

class ServerData {
  final String label;
  final double confidence;

  ServerData({required this.label, required this.confidence});

  factory ServerData.fromJson(Map<String, dynamic> data) {
    final label = data['label'];
    final confidence = data['confidence'];
    return ServerData(label: label, confidence: confidence);
  }
}

class CameraPage extends StatefulWidget {
  @override
  _CameraPageState createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  final ImagePicker _picker = ImagePicker();
  XFile? _image;

  Future<void> _takePicture() async {
    try {
      final XFile? photo = await _picker.pickImage(source: ImageSource.camera);
      if (photo != null) {
        setState(() {
          _image = photo;
        });

        await _imageToServer(photo);
      }
    } catch (e) {
      print('Error: $e');
    }
  }

  Future<void> _pickImageFromGallery() async {
    try {
      final XFile? photo = await _picker.pickImage(source: ImageSource.gallery);
      if (photo != null) {
        setState(() {
          _image = photo;
        });

        await _imageToServer(photo);
      }
    } catch (e) {
      print('Error: $e');
    }
  }

  void _confirmImageMessage(ServerData data) {
    showDialog(
        context: context,
        builder: (context) {
          return AlertDialog(
            title: Text("Is this a ${data.label}?"),
            content: _image != null
                ? Image.network(
                    _image!.path,
                    width: 300,
                    height: 300,
                    fit: BoxFit.cover,
                  )
                : Text('No image captured.'),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.pop(context); // Close dialog
                  _takePicture(); // Retake picture
                },
                child: Text('Retake'),
              ),
              TextButton(
                onPressed: () {
                  Navigator.pop(context); // Close dialog
                  _proceedWithImage(); // Proceed with current image
                },
                child: Text('Confirm'),
              ),
            ],
          );
        });
  }

  void _proceedWithImage() {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Image confirmed!')),
    );
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
        final parsedData = jsonDecode(responseData);
        final ServerData serverData = ServerData.fromJson(parsedData);

        print('Server Response: ${serverData.label} : $serverData');
        _confirmImageMessage(serverData);
      } else {
        print('Failed to upload image: ${response.statusCode}');
      }
    } catch (e) {
      print('Error sending image to server: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Camera')),
      body: Stack(
        children: [
          Center(
            child: _image == null
                ? Text('No image selected.')
                : Image.network(_image!.path, height: 300, width: 300),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16.0, vertical: 20.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  FloatingActionButton(
                    onPressed: _pickImageFromGallery,
                    heroTag: 'gallery',
                    child: Icon(Icons.photo_library),
                  ),
                  FloatingActionButton(
                    onPressed: _takePicture,
                    heroTag: 'camera',
                    child: Icon(Icons.camera_alt),
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
