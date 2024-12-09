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

class PokemonDetails {
  final String name;
  final String type;
  final double height;
  final double weight;
  final Map<String, int> stats;
  final String imageUrl;
  final String description; // New property

  PokemonDetails({
    required this.name,
    required this.type,
    required this.height,
    required this.weight,
    required this.stats,
    required this.imageUrl,
    required this.description, // Initialize the new property
  });
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

  void _showPokemonDetails(PokemonDetails details) {
    showDialog(
      context: context,
      builder: (context) {
        return Dialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16.0),
          ),
          child: Container(
            constraints: BoxConstraints(
                maxWidth: 400, maxHeight: 700), // Limit dimensions
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  padding: const EdgeInsets.all(8.0),
                  decoration: BoxDecoration(
                    color: Colors.purple.shade100,
                    borderRadius: BorderRadius.circular(16.0),
                  ),
                  child: Column(
                    children: [
                      Image.network(
                        details.imageUrl,
                        width: 100,
                        height: 100,
                        fit: BoxFit.contain,
                      ),
                      SizedBox(height: 8.0),
                      Text(
                        details.name.toUpperCase(),
                        style: TextStyle(
                          fontSize: 24.0,
                          fontWeight: FontWeight.bold,
                          color: Colors.black, // Change font color to black
                        ),
                      ),
                      Text(
                        'Type: ${details.type}',
                        style: TextStyle(
                          fontSize: 16.0,
                          color: Colors.black, // Change font color to black
                        ),
                      ),
                    ],
                  ),
                ),
                SizedBox(height: 16.0),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _buildInfoTile('Height', '${details.height} m'),
                    SizedBox(width: 16.0),
                    _buildInfoTile('Weight', '${details.weight} kg'),
                  ],
                ),
                SizedBox(height: 16.0),
                Container(
                  padding: const EdgeInsets.all(8.0),
                  decoration: BoxDecoration(
                    color: Colors.purple.shade50,
                    borderRadius: BorderRadius.circular(12.0),
                  ),
                  child: Text(
                    details.description,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 14.0,
                      color:
                          Colors.black, // Adjust text color for better contrast
                    ),
                  ),
                ),
                SizedBox(height: 16.0),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Stats:',
                      style: TextStyle(
                        fontSize: 18.0,
                        fontWeight: FontWeight.bold,
                        color: Colors.black, // Change font color to black
                      ),
                    ),
                    ...details.stats.entries.map((entry) {
                      return _buildStatRow(
                          entry.key.toUpperCase(), entry.value);
                    }).toList(),
                  ],
                ),
                SizedBox(height: 16.0),
                TextButton(
                  onPressed: () {
                    Navigator.pop(context); // Close dialog
                  },
                  child: Text(
                    'Close',
                    style: TextStyle(color: Colors.purple),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  // Helper to build individual info tiles (e.g., Height, Weight)
  Widget _buildInfoTile(String title, String value) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.purple.shade50,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.purple.shade300, width: 1),
      ),
      padding: EdgeInsets.all(8),
      child: Column(
        children: [
          Text(
            title,
            style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
                color: Colors.purple),
          ),
          Text(value, style: TextStyle(fontSize: 14, color: Colors.black)),
        ],
      ),
    );
  }

  // Helper to build individual stat rows
  Widget _buildStatRow(String statName, int statValue) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          statName.toUpperCase(),
          style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
        ),
        Text(
          "$statValue",
          style: TextStyle(fontSize: 14, color: Colors.grey.shade700),
        ),
      ],
    );
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
              onPressed: () async {
                Navigator.pop(context); // Close dialog
                await _fetchPokemonDetails(data.label.toLowerCase());

                // Show the confidence score dialog after Pokédex dialog
                _showConfidenceScoreDialog(data.confidence);
              },
              child: Text('Confirm'),
            ),
          ],
        );
      },
    );
  }

  Future<void> _fetchPokemonDetails(String pokemonName) async {
    try {
      final uri = Uri.parse('https://pokeapi.co/api/v2/pokemon/$pokemonName');
      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        final animatedSprite = data['sprites']['versions']['generation-v']
            ['black-white']['animated']['front_default'];
        final defaultSprite = data['sprites']['front_default'];

        // Fetch species for description
        final speciesUrl = data['species']['url'];
        final speciesResponse = await http.get(Uri.parse(speciesUrl));

        String description = "Description not available.";
        if (speciesResponse.statusCode == 200) {
          final speciesData = jsonDecode(speciesResponse.body);
          description = speciesData['flavor_text_entries']
              .firstWhere(
                (entry) => entry['language']['name'] == 'en',
                orElse: () => {"flavor_text": "No description available."},
              )['flavor_text']
              .replaceAll('\n', ' ')
              .replaceAll('\f', ' ');
        }

        // Update state with Pokémon details
        setState(() {
          final PokemonDetails details = PokemonDetails(
            name: data['name'],
            type: data['types'][0]['type']['name'],
            height: data['height'] / 10, // Convert decimeters to meters
            weight: data['weight'] / 10, // Convert hectograms to kg
            stats: {
              for (var stat in data['stats'])
                stat['stat']['name']: stat['base_stat'],
            },
            imageUrl: animatedSprite ?? defaultSprite,
            description: description,
          );

          // Show Pokédex dialog
          _showPokemonDetails(details);
          final confidenceValue =
              double.tryParse(details.stats['confidence']?.toString() ?? '0') ??
                  0.0;
        });
      } else {
        throw Exception("Failed to fetch Pokémon details");
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    }
  }

  void _showConfidenceScoreDialog(double confidenceScore) {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16.0),
          ),
          title: Text(
            'How Good?',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Your discovery scored: ${(confidenceScore).toStringAsFixed(2)}%!',
                style: TextStyle(fontSize: 18, color: Colors.black),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 16.0),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(context); // Close dialog
              },
              child: Text('OK', style: TextStyle(color: Colors.purple)),
            ),
          ],
        );
      },
    );
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
