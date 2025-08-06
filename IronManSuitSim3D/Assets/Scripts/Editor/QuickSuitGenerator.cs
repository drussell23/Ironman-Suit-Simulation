using UnityEngine;
using UnityEditor;
using IronManSim.Models;

namespace IronManSim.Editor
{
    public class QuickSuitGenerator
    {
        [MenuItem("GameObject/3D Object/Iron Man Suit", false, 10)]
        static void CreateIronManSuit(MenuCommand menuCommand)
        {
            // Create a new game object
            GameObject suit = new GameObject("IronManSuit_Mark85");
            
            // Add the generator component
            IronManSuitModelGenerator generator = suit.AddComponent<IronManSuitModelGenerator>();
            
            // Generate the suit
            generator.GenerateSuit();
            
            // Register the creation in the undo system
            Undo.RegisterCreatedObjectUndo(suit, "Create Iron Man Suit");
            Selection.activeObject = suit;
            
            Debug.Log("Iron Man suit created! You can find the Suit Editor under IronMan > Suit Editor menu.");
        }
        
        [MenuItem("Tools/Iron Man/Quick Generate Suit")]
        static void QuickGenerateSuit()
        {
            CreateIronManSuit(null);
        }
        
        [MenuItem("Tools/Iron Man/Open Suit Editor")]
        static void OpenSuitEditor()
        {
            IronManSuitEditorTools.ShowWindow();
        }
    }
}