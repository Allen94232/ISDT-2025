using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Collections.Generic;
using UnityEngine;

public class TCP_Lab2 : MonoBehaviour
{
    [Header("Network Settings")]
    [SerializeField] private string hostIP = "0.0.0.0";
    [SerializeField] private int port = 50555;
    TcpListener server = null;
    TcpClient client = null;
    NetworkStream stream = null;
    Thread thread;

    [Serializable]
    public class Message
    {
        public string some_string;
        public int id;
        public Vector3[] ArUcoCornerPos;
        public Quaternion rotation;
        public Vector3 transformed_position;
    }

    private static object Lock = new object();
    private List<Message> MessageQue = new List<Message>();

    // 改成 Dictionary 儲存 ID 對應的位置
    private Dictionary<int, Vector3> transformedPositions = new Dictionary<int, Vector3>();

    private void Start()
    {
        thread = new Thread(SetupServer) { IsBackground = true };
        thread.Start();
    }

    private void Update()
    {
        // 處理收到的訊息
        lock (Lock)
        {
            foreach (Message msg in MessageQue)
            {
                Debug.Log($"Received: ID={msg.id}, Pos={msg.ArUcoCornerPos?.Length}, Rot={msg.rotation}, Transformed Pos={msg.transformed_position}");

                // 更新該 ID 的 transformed_position
                transformedPositions[msg.id] = msg.transformed_position;
            }
            MessageQue.Clear();
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

            while (true)
            {
                Debug.Log("Waiting for connection...");
                client = server.AcceptTcpClient();
                Debug.Log("Connected!");

                stream = client.GetStream();
                string receiveBuffer = "";

                int bytesRead;
                while ((bytesRead = stream.Read(buffer, 0, buffer.Length)) != 0)
                {
                    receiveBuffer += Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    int newlineIndex;
                    while ((newlineIndex = receiveBuffer.IndexOf('\n')) >= 0)
                    {
                        string line = receiveBuffer.Substring(0, newlineIndex).Trim();
                        receiveBuffer = receiveBuffer.Substring(newlineIndex + 1);
                        if (line.Length == 0)
                        {
                            continue;
                        }

                        try
                        {
                            Message message = Decode(line);
                            lock (Lock)
                            {
                                MessageQue.Add(message);
                            }
                        }
                        catch (Exception decodeError)
                        {
                            Debug.LogWarning("Invalid JSON message: " + decodeError.Message);
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
            server?.Stop();
        }
    }

    private void OnApplicationQuit()
    {
        stream?.Close();
        client?.Close();
        server?.Stop();
        if (thread != null && thread.IsAlive)
        {
            thread.Join(500);
        }
    }

    public void SendMessageToClient(Message message)
    {
        if (stream == null || !stream.CanWrite)
        {
            Debug.LogWarning("TCP client is not connected.");
            return;
        }

        byte[] msg = Encoding.UTF8.GetBytes(Encode(message) + "\n");
        stream.Write(msg, 0, msg.Length);
        Debug.Log("Sent: " + message);
    }

    public string Encode(Message message)
    {
        return JsonUtility.ToJson(message);
    }

    public Message Decode(string json_string)
    {
        return JsonUtility.FromJson<Message>(json_string);
    }

    public void CreateAndSendSpatialAnchorData(int id, Vector3 centerPos, Quaternion rot, float ArUcoSize)
    {
        Message msg = new Message();
        msg.some_string = "From Server";
        msg.id = id;

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

        msg.rotation = rot;
        SendMessageToClient(msg);
    }

    // 安全取得 transformed_position
    public Vector3 SetTransformedPosition(int id)
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
}
