import 'package:firebase_ui_auth/firebase_ui_auth.dart';
import 'package:flutter/material.dart';
import 'package:google_nav_bar/google_nav_bar.dart';
import 'package:camera/camera.dart';
import 'imagepick.dart';

class HomeScreen extends StatefulWidget {
    const HomeScreen({super.key});

    @override
    State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
    int _selectedIndex = 0;

    final List<Widget> _pages = [
        Center(child:Text("Home Page")),
        CameraPage(),
        Center(child:Text("Leaderboard Page")),
        ProfileScreen(
            appBar: AppBar(
                title:  const Text('User Profile'),
            ),
            actions: [
                SignedOutAction((context) {
                    Navigator.of(context).pop();
                })
            ],
        ),
    ];

    @override
    Widget build(BuildContext context) {
        return Scaffold(
            body: _pages[_selectedIndex],
            bottomNavigationBar: GNav(
                backgroundColor: Colors.red,
                color: Colors.white,
                activeColor: Colors.white,
                tabBackgroundColor: const Color.fromARGB(255, 44, 33, 33),
                gap: 8,
                padding: const EdgeInsets.all(16),
                tabs: [
                    const GButton(icon: Icons.home,
                        text: 'Home',
                    ),

                    const GButton(icon: Icons.camera,
                        text: 'camera',
                    ),

                    const GButton(icon: Icons.leaderboard,
                        text: 'Leaderboard',
                    ),

                    //Button and page for Settings
                    GButton(icon: Icons.settings,
                        text: 'settings',
                    ),
                ],
                selectedIndex: _selectedIndex,
                onTabChange: (index) {
                    setState(() {
                        _selectedIndex = index;
                    });
                }
            ),
    );
  }
  }
  
