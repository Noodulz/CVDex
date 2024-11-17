import 'package:firebase_ui_auth/firebase_ui_auth.dart';
import 'package:flutter/material.dart';
import 'package:google_nav_bar/google_nav_bar.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
 State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
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

         const GButton(icon: Icons.favorite_border,
          text: 'likes',
        ),
        
         const GButton(icon: Icons.leaderboard,
          text: 'Leaderboard',
        ),
        
        //Button and page for Settings
         GButton(icon: Icons.settings,
          text: 'settings',
          onPressed: () {
            Navigator.push(
              context,
               MaterialPageRoute<ProfileScreen> (
                builder: (context) => ProfileScreen(
                  appBar: AppBar(
                    title:  const Text('User Profile'),
                  ),
                  actions: [
                    SignedOutAction((context) {
                      Navigator.of(context).pop();
                    })
                  ],
                ),
               ),
            );
          }
        ),
        ]
      ),
    );
  }
  }
  