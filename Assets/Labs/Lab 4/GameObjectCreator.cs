using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static OVRInput;

public class GameObjectCreator : MonoBehaviour
{
    [SerializeField] private Controller controller;
    [SerializeField] private TCP_Lab4 server;
    [SerializeField] private float ArUcoSize = 0.16f;
    [SerializeField] private float timeoutDuration = 2.0f; // 2���W��

    public GameObject instruction_1;
    public GameObject instruction_2;
    public GameObject instruction_3;

    // Controller movement settings
    [SerializeField] private float moveSpeed = 1.0f;
    [SerializeField] private float rotateSpeed = 90.0f;

    // Public list for different prefabs based on ID
    public List<GameObject> prefabsByID = new List<GameObject>();
    
    // ArUco marker IDs mapping
    private int[] arucoIDs = { 3, 5, 10 }; // ArUco marker IDs to use
    private int currentIDIndex = 0; // Index into arucoIDs array
    public List<GameObject> createdObjects = new List<GameObject>();

    // �l�̫ܳ��s�ɶ�
    private Dictionary<int, DateTime> lastUpdateTime = new Dictionary<int, DateTime>();

    void Start()
    {
        if (prefabsByID.Count < 3)
        {
            Debug.LogError("Please assign at least 3 prefabs in the inspector for IDs 3, 5, and 10.");
        }
        instruction_1.SetActive(false);
        instruction_2.SetActive(false);
        instruction_3.SetActive(false);
    }

    void Update()
    {
        if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, controller) && !Lab4_GameManager.instance.isGaming && server.isCalibrated)
        {
            CreateGameObject();
        }

        // Handle instruction quest3 movement when not calibrated and instruction is active
        if (!Lab4_GameManager.instance.isGaming)
        {
            HandleInstructionMovement();
        }

        // Update positions based on transformed positions from server
        foreach (GameObject obj in createdObjects)
        {
            if (obj != null)
            {
                int id = obj.GetComponent<ObjectID>().id;
                Vector3 newpos = server.GetTransformedPosition(id);
                DateTime lastUpdate = server.GetLastUpdateTime(id);

                if (newpos != Vector3.zero)
                {
                    // �ˬd�O�_�b�W�ɽd��
                    if ((DateTime.Now - lastUpdate).TotalSeconds <= timeoutDuration)
                    {
                        // Update entire object position with offset compensation
                        ObjectID objID = obj.GetComponent<ObjectID>();
                        
                        if (obj.transform.childCount > 0 && objID != null)
                        {
                            // 使用本地座標偏移，考慮物件的旋轉
                            // 將本地偏移量轉換為世界座標偏移量
                            Vector3 worldOffset = obj.transform.TransformVector(objID.childOffset);
                            Vector3 newRootPos = newpos - worldOffset;
                            
                            obj.transform.position = newRootPos;
                            
                            // Verify child position
                            Transform child = obj.transform.GetChild(0);
                            Debug.Log($"Updated root position for ID {id}: {newRootPos}");
                            Debug.Log($"Child(0) position is now: {child.position} (target was: {newpos})");
                            Debug.Log($"Local offset: {objID.childOffset}, World offset: {worldOffset}");
                        }
                        else
                        {
                            obj.transform.position = newpos;
                            Debug.Log($"Updated root position for ID {id}: {newpos}");
                        }

                        if (createdObjects.IndexOf(obj) == 0)
                        {
                            if(!obj.activeSelf && Lab4_GameManager.instance.isGaming)
                            {
                                Lab4_GameManager.instance.ToggleExchanging(true);
                            }
                        }

                            // if third index in createdObjects
                        if (createdObjects.IndexOf(obj) == 2)
                        {
                            // check second index in createdObjects is active
                            if (createdObjects.Count > 1 && createdObjects[1] != null && createdObjects[1].activeSelf)
                            {
                                //obj.SetActive(false);
                                obj.SetActive(true);
                            }
                            else
                            {
                                obj.SetActive(true);
                                //obj.SetActive(false);
                            }
                        }

                        else
                        {
                            obj.SetActive(true);
                        }
                    }
                    else
                    {
                        if (createdObjects.IndexOf(obj) == 0)
                        {
                            if(obj.activeSelf && Lab4_GameManager.instance.isGaming)
                            {
                                Lab4_GameManager.instance.ToggleExchanging(false);
                            }
                        }

                        // �W�ɡA���éμаO�����i�a
                        obj.SetActive(false);
                        Debug.Log($"ArUco {id} timed out, hiding object");
                    }
                }
            }
        }
    }

    private void HandleInstructionMovement()
    {
        Transform activeInstruction = GetActiveInstruction();
        if (activeInstruction == null) return;

        // Get left and right controller inputs
        Vector2 leftThumbstick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.LTouch);
        Vector2 rightThumbstick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.RTouch);

        // Left controller for movement (X/Z plane)
        if (leftThumbstick.magnitude > 0.1f)
        {
            Vector3 moveDirection = new Vector3(leftThumbstick.x, 0, leftThumbstick.y);
            activeInstruction.position += moveDirection * moveSpeed * Time.deltaTime;
        }

        // Right controller for rotation
        if (rightThumbstick.magnitude > 0.1f)
        {
            // Horizontal movement for Y-axis rotation
            float yRotation = rightThumbstick.x * rotateSpeed * Time.deltaTime;
            activeInstruction.Rotate(0, yRotation, 0, Space.World);

            // Vertical movement for X-axis rotation
            float xRotation = -rightThumbstick.y * rotateSpeed * Time.deltaTime;
            activeInstruction.Rotate(xRotation, 0, 0, Space.World);
        }

        // Alternative: Use trigger buttons for Y movement
        if (OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.LTouch))
        {
            activeInstruction.position += Vector3.up * moveSpeed * Time.deltaTime;
        }
        if (OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch))
        {
            activeInstruction.position += Vector3.down * moveSpeed * Time.deltaTime;
        }
    }

    private Transform GetActiveInstruction()
    {
        if (instruction_1.activeSelf && instruction_1.transform.childCount > 1)
        {
            return instruction_1.transform.GetChild(1); // quest3 is the second child (index 1)
        }
        else if (instruction_2.activeSelf && instruction_2.transform.childCount > 1)
        {
            return instruction_2.transform.GetChild(1);
        }
        else if (instruction_3.activeSelf && instruction_3.transform.childCount > 1)
        {
            return instruction_3.transform.GetChild(1);
        }

        return null; // No active instruction or quest3 not found
    }

    public void CreateGameObject()
    {
        // Ensure we don't exceed the number of available ArUco IDs
        if (currentIDIndex >= arucoIDs.Length || currentIDIndex >= prefabsByID.Count)
        {
            Debug.LogWarning("No more ArUco IDs or prefabs available. Index: " + currentIDIndex);
            return;
        }

        // Get current ArUco ID and corresponding prefab
        int currentID = arucoIDs[currentIDIndex];
        GameObject prefabToUse = prefabsByID[currentIDIndex];

        if (prefabToUse == null)
        {
            Debug.LogError("Prefab for ArUco ID " + currentID + " is null!");
            return;
        }

        // Create the object at controller position
        GameObject newObject = Instantiate(prefabToUse, 
                                         OVRInput.GetLocalControllerPosition(controller),
                                         OVRInput.GetLocalControllerRotation(controller));

        // Add ObjectID component to track the ID and child offset
        ObjectID idComp = newObject.AddComponent<ObjectID>();
        idComp.id = currentID;
        
        // 計算並儲存本地座標偏移量（不受旋轉影響）
        if (newObject.transform.childCount > 0)
        {
            Transform child = newObject.transform.GetChild(0);
            // 使用 InverseTransformVector 將世界座標偏移轉換為本地座標偏移
            Vector3 worldOffset = child.position - newObject.transform.position;
            idComp.childOffset = newObject.transform.InverseTransformVector(worldOffset);
            Debug.Log($"Stored LOCAL child offset for ID {currentID}: {idComp.childOffset}");
        }
        else
        {
            idComp.childOffset = Vector3.zero;
        }

        // Add to created objects list
        createdObjects.Add(newObject);

        // Send spatial anchor data to server using child(0) position
        if (server != null)
        {
            // Use child(0) position if it exists, otherwise use root position
            Vector3 anchorPosition;
            Quaternion anchorRotation;
            
            if (newObject.transform.childCount > 0)
            {
                Transform child = newObject.transform.GetChild(0);
                anchorPosition = child.position;
                anchorRotation = child.rotation;
                Debug.Log($"Using child(0) position for ID {currentID}: {anchorPosition}");
            }
            else
            {
                anchorPosition = newObject.transform.position;
                anchorRotation = newObject.transform.rotation;
                Debug.Log($"No child found, using root position for ID {currentID}: {anchorPosition}");
            }
            
            server.CreateAndSendGameObjectData(currentID, 
                                                anchorPosition, 
                                                anchorRotation, 
                                                ArUcoSize);
        }

        Debug.Log($"Created object with ArUco ID: {currentID}");
        Debug.Log($"Object has {newObject.transform.childCount} children");
        if (newObject.transform.childCount > 0)
        {
            Debug.Log($"Child(0) name: {newObject.transform.GetChild(0).name}");
        }

        // Increment index for next object (max 3 objects for 3 ArUco markers)
        currentIDIndex++;

        // Show corresponding instruction based on index
        if (currentIDIndex == 1) // After creating first object (ID 3)
        {
            instruction_1.SetActive(false);
            instruction_2.SetActive(true);
            instruction_3.SetActive(false);
        }
        else if (currentIDIndex == 2) // After creating second object (ID 5)
        {
            instruction_1.SetActive(false);
            instruction_2.SetActive(false);
            instruction_3.SetActive(true);
        }
        else if (currentIDIndex >= 3) // After creating third object (ID 10)
        {
            instruction_1.SetActive(false);
            instruction_2.SetActive(false);
            instruction_3.SetActive(false);
        }
    }
}

// Component to store object ID and child offset
public class ObjectID : MonoBehaviour
{
    public int id;
    public Vector3 childOffset; // Offset from root to child(0) at creation time
}