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

  Future<Map<String, dynamic>?> fetchPokemonData(String pokemonName) async {
    final url = Uri.parse(
        'https://pokeapi.co/api/v2/pokemon/${pokemonName.toLowerCase()}');
    try {
      final response = await http.get(url);

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        print('Error: Pokémon not found');
        return null; // Handle not found Pokémon
      }
    } catch (e) {
      print('Error fetching Pokémon data: $e');
      return null;
    }
  }

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
                  _proceedWithImage(data.label); // Proceed with current image
                },
                child: Text('Confirm'),
              ),
            ],
          );
        });
  }

  void _showPokedexPopup(Map<String, dynamic> pokemonData) {
    final imageUrl = pokemonData['sprites']['front_default'] ?? '';
    final name = pokemonData['name'];
    final height = pokemonData['height'] / 10; // Convert to meters
    final weight = pokemonData['weight'] / 10; // Convert to kilograms
    final types = (pokemonData['types'] as List)
        .map((typeInfo) => typeInfo['type']['name'])
        .join(', ');
    final stats = (pokemonData['stats'] as List)
        .map((stat) => "${stat['stat']['name']}: ${stat['base_stat']}")
        .join('\n');

    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Pokémon Image
              if (imageUrl.isNotEmpty)
                Image.network(imageUrl, height: 150, fit: BoxFit.cover),
              SizedBox(height: 10),

              // Pokémon Name
              Text(
                name.toUpperCase(),
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 10),

              // Pokémon Details
              Text('Type: $types', style: TextStyle(fontSize: 16)),
              Text('Height: ${height.toStringAsFixed(1)} m',
                  style: TextStyle(fontSize: 16)),
              Text('Weight: ${weight.toStringAsFixed(1)} kg',
                  style: TextStyle(fontSize: 16)),
              SizedBox(height: 10),

              // Pokémon Stats
              Text(
                'Stats:\n$stats',
                style: TextStyle(fontSize: 14),
                textAlign: TextAlign.center,
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(context); // Close the dialog
              },
              child: Text('Close'),
            ),
          ],
        );
      },
    );
  }

  void _proceedWithImage(String pokemonName) async {
    final pokemonData = await fetchPokemonData(pokemonName);

    if (pokemonData != null) {
      _showPokedexPopup(pokemonData);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Pokémon not found in PokéAPI')),
      );
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
