/*
Reference
Implementing a Basic TCP Server in Unity: A Step-by-Step Guide
By RabeeQiblawi Nov 20, 2023
https://medium.com/@rabeeqiblawi/implementing-a-basic-tcp-server-in-unity-a-step-by-Step-guide-449d8504d1c5
*/

using System;
using System.Collections.Generic;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public class TCP_Lab4 : MonoBehaviour
{
    const string hostIP = "10.47.102.7"; // Select your IP
    const int port = 143; // Select your port
    TcpListener server = null;
    TcpClient client = null;
    NetworkStream stream = null;
    Thread thread;

    public Transform LHand;
    public Transform RHand;
    public Transform Head;

    public GameObject OculusCamera;
    //public GameObject MainCamera;
    public GameObject Avatar;

    public Transform HeadIKTarget;
    public Transform LeftHandIKTarget;
    public Transform RightHandIKTarget;

    // Combined Message class from both Lab2 and Lab3
    [Serializable]
    public class Message
    {
        // Lab3 fields for avatar IK
        public float LHand_x;
        public float LHand_y;
        public float LHand_z;
        public float RHand_x;
        public float RHand_y;
        public float RHand_z;
        public float Head_x;
        public float Head_y;
        public float Head_z;
        
        // Lab2 fields for ArUco - Lab2 Complete style with center point
        public string some_string;
        public int id;
        public Vector3 center_position;  // Send center position for calibration
        public Vector3[] ArUcoCornerPos; // Keep for compatibility, but primarily use center
        public Quaternion rotation;
        public Vector3 transformed_position;
    }

    public bool isCalibrated = false;
    private bool allArUcoPositioned = false;
    public HashSet<int> positionedArUcoIds = new HashSet<int>();
    private const int TOTAL_ARUCO_COUNT = 3;

    private static object Lock = new object();
    private List<Message> MessageQue = new List<Message>();
    
    // TCP message buffering
    private string messageBuffer = "";
    private const int MAX_BUFFER_SIZE = 8192;

    // Dictionary to store ID corresponding positions
    private Dictionary<int, Vector3> transformedPositions = new Dictionary<int, Vector3>();

    // Reference to GameObjectCreator
    public GameObjectCreator gameObjectCreator;

    // 添加檢測狀態追蹤
    private Dictionary<int, DateTime> lastArUcoUpdateTime = new Dictionary<int, DateTime>();
    private float detectionTimeout = 1.0f; // 1秒超時

    private void Start()
    {
        thread = new Thread(new ThreadStart(SetupServer));
        thread.Start();
    }

    private void Update()
    {
        // Process received messages
        lock (Lock)
        {
            foreach (Message msg in MessageQue)
            {
                // Handle avatar IK messages
                if (isCalibrated && msg.id == 0) // ID 0 for avatar data
                {
                    if (Lab4_GameManager.instance.isExchanging || !Lab4_GameManager.instance.isGaming)
                        Move(msg);
                }
                // Handle ArUco messages
                else if (msg.id > 0)
                {
                    Debug.Log($"Received: ID={msg.id}, Center={msg.center_position}, Corners={msg.ArUcoCornerPos?.Length}, Transformed Pos={msg.transformed_position}");
                    
                    // Update the transformed_position for the ID
                    transformedPositions[msg.id] = msg.transformed_position;
                    lastArUcoUpdateTime[msg.id] = DateTime.Now; // 記錄更新時間
                    positionedArUcoIds.Add(msg.id);
                    
                    // Check if all ArUco markers are positioned
                    CheckAllArUcoPositioned();
                }
            }
            MessageQue.Clear();
        }

        // Press Space button to calibrate avatar (same as Lab3)
        if (Input.GetKeyDown(KeyCode.Space) && !isCalibrated)
        {
            CalibrateAvatar();
        }
    }

    private void SetupServer()
    {
        try
        {
            IPAddress localAddr = IPAddress.Parse(hostIP);
            server = new TcpListener(localAddr, port);
            server.Start();

            byte[] buffer = new byte[1024];
            string data = null;

            while (true)
            {
                Debug.Log("Waiting for connection...");
                client = server.AcceptTcpClient();
                Debug.Log("Connected!");

                data = null;
                stream = client.GetStream();
                
                // Clear message buffer for new connection
                messageBuffer = "";

                int i;
                while ((i = stream.Read(buffer, 0, buffer.Length)) != 0)
                {
                    data = Encoding.UTF8.GetString(buffer, 0, i);
                    
                    // Process received data with buffering
                    List<Message> messages = ProcessReceivedData(data);
                    
                    lock (Lock)
                    {
                        foreach (Message msg in messages)
                        {
                            MessageQue.Add(msg);
                        }
                    }
                }
                client.Close();
            }
        }
        catch (SocketException e)
        {
            Debug.Log("SocketException: " + e);
        }
        finally
        {
            server.Stop();
        }
    }

    private void OnApplicationQuit()
    {
        stream?.Close();
        client?.Close();
        server?.Stop();
        thread?.Abort();
    }

    public void SendMessageToClient(Message message)
    {
        if (stream != null && !Lab4_GameManager.instance.isGaming)
        {
            try
            {
                string jsonMessage = Encode(message);
                
                // Add newline separator to help with message boundary detection
                string messageWithSeparator = jsonMessage + "\n";
                byte[] msg = Encoding.UTF8.GetBytes(messageWithSeparator);
                
                stream.Write(msg, 0, msg.Length);
                stream.Flush(); // Force immediate send
                
                // More concise logging
                if (message.id == 0)
                {
                    Debug.Log("Sent Avatar Data");
                }
                else
                {
                    Debug.Log($"Sent ArUco Data (ID: {message.id})");
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Failed to send message: {e.Message}");
            }
        }
    }

    public string Encode(Message message)
    {
        return JsonUtility.ToJson(message, true);
    }

    public Message Decode(string json_string)
    {
        try
        {
            return JsonUtility.FromJson<Message>(json_string);
        }
        catch (System.ArgumentException e)
        {
            Debug.LogError($"JSON Parse Error: {e.Message}");
            Debug.LogError($"Problematic JSON: {json_string}");
            return null;
        }
    }
    
    private List<Message> ProcessReceivedData(string newData)
    {
        List<Message> messages = new List<Message>();
        
        // Add new data to buffer
        messageBuffer += newData;
        
        // Extract complete JSON messages from buffer
        while (true)
        {
            // Find the start of a JSON object
            int startIdx = messageBuffer.IndexOf('{');
            if (startIdx == -1)
            {
                // No JSON start found, clear buffer of any garbage
                messageBuffer = "";
                break;
            }
            
            // Remove any data before the JSON start
            if (startIdx > 0)
            {
                messageBuffer = messageBuffer.Substring(startIdx);
            }
            
            // Find complete JSON object using brace counting
            int braceCount = 0;
            int endIdx = -1;
            
            for (int i = 0; i < messageBuffer.Length; i++)
            {
                char c = messageBuffer[i];
                if (c == '{')
                {
                    braceCount++;
                }
                else if (c == '}')
                {
                    braceCount--;
                    if (braceCount == 0)
                    {
                        endIdx = i + 1;
                        break;
                    }
                }
            }
            
            if (endIdx == -1)
            {
                // Incomplete JSON, wait for more data
                break;
            }
            
            // Extract complete JSON
            string jsonStr = messageBuffer.Substring(0, endIdx);
            messageBuffer = messageBuffer.Substring(endIdx);
            
            // Try to decode the JSON
            Message message = Decode(jsonStr);
            if (message != null)
            {
                messages.Add(message);
                Debug.Log($"Successfully parsed message: ID={message.id}");
            }
            
            // Clean up buffer if it gets too large
            if (messageBuffer.Length > MAX_BUFFER_SIZE)
            {
                Debug.LogWarning($"Buffer too large ({messageBuffer.Length} chars), clearing...");
                messageBuffer = "";
                break;
            }
        }
        
        return messages;
    }

    // Avatar movement method from Lab3
    public void Move(Message message)
    {
        LHand.localPosition = new Vector3(message.LHand_x, message.LHand_y, message.LHand_z);
        RHand.localPosition = new Vector3(message.RHand_x, message.RHand_y, message.RHand_z);
        Head.localPosition = new Vector3(message.Head_x, message.Head_y, message.Head_z);
        Debug.Log("Left Hand: " + LHand.position.ToString());
        Debug.Log("Right Hand: " + RHand.position.ToString());
        Debug.Log("Head: " + Head.position.ToString());
    }

    // Calibrate avatar method from Lab3
    private void CalibrateAvatar()
    {
        // Read IK target positions to message
        Message msg = new Message();
        msg.id = 0; // Special ID for avatar data
        msg.LHand_x = LeftHandIKTarget.position.x;
        msg.LHand_y = LeftHandIKTarget.position.y;
        msg.LHand_z = LeftHandIKTarget.position.z;
        msg.RHand_x = RightHandIKTarget.position.x;
        msg.RHand_y = RightHandIKTarget.position.y;
        msg.RHand_z = RightHandIKTarget.position.z;
        msg.Head_x = HeadIKTarget.position.x;
        msg.Head_y = HeadIKTarget.position.y;
        msg.Head_z = HeadIKTarget.position.z;

        SendMessageToClient(msg);

        Avatar.GetComponent<IKTargetFollowVRRig>().head.vrTarget = Head;
        Avatar.GetComponent<IKTargetFollowVRRig>().leftHand.vrTarget = LHand;
        Avatar.GetComponent<IKTargetFollowVRRig>().rightHand.vrTarget = RHand;

        isCalibrated = true;

        gameObjectCreator.instruction_1.SetActive(true);

        /*
        // Set main camera position to in front of oculus camera position 2 meters and face to oculus camera
        MainCamera.transform.position = OculusCamera.transform.position + OculusCamera.transform.forward * 2.0f;
        MainCamera.transform.position = new Vector3(MainCamera.transform.position.x, 1.0f, MainCamera.transform.position.z);

        MainCamera.transform.rotation = Quaternion.Euler(0, OculusCamera.transform.eulerAngles.y + 180.0f, 0);

        OculusCamera.SetActive(false);
        MainCamera.SetActive(true);
        */
    }

    // Create and send spatial anchor data method - Lab2 Complete style with center point emphasis
    public void CreateAndSendGameObjectData(int id, Vector3 centerPos, Quaternion rot, float ArUcoSize)
    {
        if (!Lab4_GameManager.instance.isGaming)
        {
            Message msg = new Message();
            msg.some_string = "From Server";
            msg.id = id;

            // Primary: Send center position (Lab2 Complete style for accurate calibration)
            msg.center_position = centerPos;
            msg.rotation = rot;
            
            // Secondary: Still calculate corners for compatibility, but center is primary
            float halfSize = ArUcoSize / 2f;
            Vector3[] localCorners = new Vector3[]
            {
                new Vector3(-halfSize, 0,  halfSize),
                new Vector3( halfSize, 0,  halfSize),
                new Vector3( halfSize, 0, -halfSize),
                new Vector3(-halfSize, 0, -halfSize)
            };

            msg.ArUcoCornerPos = new Vector3[4];
            for (int i = 0; i < 4; i++)
            {
                msg.ArUcoCornerPos[i] = centerPos + rot * localCorners[i];
            }
            
            SendMessageToClient(msg);
            
            Debug.Log($"Sent ArUco anchor data: ID={id}, Center={centerPos}");
        }
    }

    // Safely get transformed_position
    public Vector3 GetTransformedPosition(int id)
    {
        if (transformedPositions.TryGetValue(id, out Vector3 pos))
        {
            return pos;
        }
        else
        {
            return Vector3.zero;
        }
    }

    // Check if all ArUco markers are positioned
    private void CheckAllArUcoPositioned()
    {
        if (!allArUcoPositioned && positionedArUcoIds.Count >= TOTAL_ARUCO_COUNT && isCalibrated)
        {
            // if press Space button, start the game
            if (Input.GetKeyDown(KeyCode.Space))
            {
                if (!Lab4_GameManager.instance.isGaming)
                {
                    allArUcoPositioned = true;
                    Lab4_GameManager.instance.isGaming = true;
                    Lab4_GameManager.instance.isExchanging = false;

                    Lab4_GameManager.instance.gameUI = gameObjectCreator.createdObjects[0].transform.GetChild(2).gameObject;

                    // set all second child of createdObjects renderer to false
                    foreach (GameObject obj in gameObjectCreator.createdObjects)
                    {
                        if (obj.transform.childCount > 1)
                        {
                            // set all renderers in the first child of second child to false
                            Renderer[] renderers = obj.transform.GetChild(1).GetChild(0).GetComponentsInChildren<Renderer>();
                            foreach (Renderer rend in renderers)
                            {
                                rend.enabled = false;
                            }
                        }
                    }

                    Debug.Log("All ArUco markers positioned! Game started.");
                }
            }
        }
    }

    // 新增方法：檢查 ArUco 是否當前被檢測到
    public bool IsArUcoCurrentlyDetected(int id)
    {
        if (lastArUcoUpdateTime.TryGetValue(id, out DateTime lastUpdate))
        {
            return (DateTime.Now - lastUpdate).TotalSeconds <= detectionTimeout;
        }
        return false;
    }

    // 新增方法：獲取最後更新時間
    public DateTime GetLastUpdateTime(int id)
    {
        if (lastArUcoUpdateTime.TryGetValue(id, out DateTime lastUpdate))
        {
            return lastUpdate;
        }
        return DateTime.MinValue;
    }
}