using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Collections.Generic;
using UnityEngine;

public class TCP_Lab2 : MonoBehaviour
{
    const string hostIP = "10.47.101.196"; // Select your IP
    const int port = 143; // Select your port
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
        thread = new Thread(new ThreadStart(SetupServer));
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
            string data = null;

            while (true)
            {
                Debug.Log("Waiting for connection...");
                client = server.AcceptTcpClient();
                Debug.Log("Connected!");

                data = null;
                stream = client.GetStream();

                int i;
                while ((i = stream.Read(buffer, 0, buffer.Length)) != 0)
                {
                    data = Encoding.UTF8.GetString(buffer, 0, i);
                    Message message = Decode(data);
                    lock (Lock)
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
        stream?.Close();
        client?.Close();
        server?.Stop();
        thread?.Abort();
    }

    public void SendMessageToClient(Message message)
    {
        byte[] msg = Encoding.UTF8.GetBytes(Encode(message));
        stream.Write(msg, 0, msg.Length);
        Debug.Log("Sent: " + message);
    }

    public string Encode(Message message)
    {
        return JsonUtility.ToJson(message, true);
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
