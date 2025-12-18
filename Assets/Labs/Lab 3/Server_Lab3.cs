/*
Reference
Implementing a Basic TCP Server in Unity: A Step-by-Step Guide
By RabeeQiblawi Nov 20, 2023
https://medium.com/@rabeeqiblawi/implementing-a-basic-tcp-server-in-unity-a-step-by-step-guide-449d8504d1c5
*/

using System;
using System.Collections.Generic;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public class TCP_Lab3 : MonoBehaviour
{
    const string hostIP = "10.47.101.179"; // Select your IP
    const int port = 143; // Select your port
    TcpListener server = null;
    TcpClient client = null;
    NetworkStream stream = null;
    Thread thread;

    public Transform LHand;
    public Transform RHand;
    public Transform Head;

    public GameObject OculusCamera;
    public GameObject MainCamera;

    // Define your own message
    [Serializable]
    public class Message
    {
        public float LHand_x;
        public float LHand_y;
        public float LHand_z;
        public float RHand_x;
        public float RHand_y;
        public float RHand_z;
        public float Head_x;
        public float Head_y;
        public float Head_z;
    }

    public GameObject Avatar;

    public Transform HeadIKTarget;
    public Transform LeftHandIKTarget;
    public Transform RightHandIKTarget;

    public bool isCalibrated = false;

    private float timer = 0;
    private static object Lock = new object();
    private List<Message> MessageQue = new List<Message>();


    private void Start()
    {
        thread = new Thread(new ThreadStart(SetupServer));
        thread.Start();
    }

    private void Update()
    {
        lock(Lock)
        {
            foreach (Message msg in MessageQue)
            {
                if (isCalibrated)
                    Move(msg);
            }
            MessageQue.Clear();
        }

        // if press Space button, set isCalibrated to true
        if (Input.GetKeyDown(KeyCode.Space) && !isCalibrated)
        {
            // read IK target positions to message
            Message msg = new Message();
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

            // set main camera position to in front of oculus camera position 5 meters and face to oculus camera
            MainCamera.transform.position = OculusCamera.transform.position + OculusCamera.transform.forward * 2.0f;
            MainCamera.transform.position = new Vector3(MainCamera.transform.position.x, 1.0f, MainCamera.transform.position.z);

            MainCamera.transform.rotation = Quaternion.Euler(0, OculusCamera.transform.eulerAngles.y + 180.0f , 0);

            OculusCamera.SetActive(false);
            MainCamera.SetActive(true);
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

                // Receive message from client    
                int i;
                while ((i = stream.Read(buffer, 0, buffer.Length)) != 0)
                {
                    data = Encoding.UTF8.GetString(buffer, 0, i);
                    Message message = Decode(data);
                    Debug.Log(message.ToString());
                    lock(Lock)
                    {
                        MessageQue.Add(message);
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
        stream.Close();
        client.Close();
        server.Stop();
        thread.Abort();
    }

    public void SendMessageToClient(Message message)
    {
        byte[] msg = Encoding.UTF8.GetBytes(Encode(message));
        stream.Write(msg, 0, msg.Length);
        Debug.Log("Sent: " + message);
    }

    // Encode message from struct to Json String
    public string Encode(Message message)
    {
        return JsonUtility.ToJson(message, true);
    }

    // Decode messaage from Json String to struct
    public Message Decode(string json_string)
    {
        Message msg = JsonUtility.FromJson<Message>(json_string);
        return msg;
    }

    public void Move(Message message)
    {
        LHand.localPosition = new Vector3(message.LHand_x, message.LHand_y, message.LHand_z);
        RHand.localPosition = new Vector3(message.RHand_x, message.RHand_y, message.RHand_z);
        Head.localPosition = new Vector3(message.Head_x, message.Head_y, message.Head_z);
        Debug.Log("Left Hand: " + LHand.position.ToString());
        Debug.Log("Right Hand: " + RHand.position.ToString());
        Debug.Log("Head: " + Head.position.ToString());
    }
}
