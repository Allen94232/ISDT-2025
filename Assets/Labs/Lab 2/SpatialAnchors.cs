using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static OVRInput;
using TMPro;

public class SpatialAnchors : MonoBehaviour
{
    [SerializeField] private Controller controller;
    [SerializeField] private TCP_Lab2 server;
    [SerializeField] private float ArUcoSize = 0.16f;
    private int count = 0;

    public GameObject anchorPrefab;
    public List<GameObject> createdAnchors = new List<GameObject>();
    private Canvas canvas;
    private TextMeshProUGUI idText;
    private TextMeshProUGUI positionText;

    void Update()
    {
        if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, controller))
        {
            CreateSpatialAnchor();
        }

        // 根據 anchor 的 ID 來更新位置
        foreach (GameObject anchor in createdAnchors)
        {
            if (anchor != null)
            {
                int id = anchor.GetComponent<AnchorID>().id; // 從元件讀取 ID
                Vector3 newpos = server.SetTransformedPosition(id);

                if (newpos != Vector3.zero)
                {
                    anchor.transform.GetChild(0).position = newpos;
                }

                positionText = anchor.GetComponentInChildren<Canvas>().transform.GetChild(1).GetComponent<TextMeshProUGUI>();
                positionText.text = anchor.transform.GetChild(0).position.ToString();
            }
        }
    }

    public void CreateSpatialAnchor()
    {
        GameObject anchor = Instantiate(anchorPrefab, OVRInput.GetLocalControllerPosition(controller),
                                                     OVRInput.GetLocalControllerRotation(controller));

        canvas = anchor.GetComponentInChildren<Canvas>();

        // 用 AnchorID script 記錄 ID
        AnchorID idComp = anchor.AddComponent<AnchorID>();
        idComp.id = ++count;

        idText = canvas.transform.GetChild(0).GetComponent<TextMeshProUGUI>();
        idText.text = "ID: " + count;

        positionText = canvas.transform.GetChild(1).GetComponent<TextMeshProUGUI>();
        positionText.text = anchor.transform.GetChild(0).GetChild(0).position.ToString();

        anchor.AddComponent<OVRSpatialAnchor>();

        createdAnchors.Add(anchor);

        if (server != null)
        {
            server.CreateAndSendSpatialAnchorData(count, anchor.transform.GetChild(0).position, anchor.transform.GetChild(0).rotation, ArUcoSize);
        }
    }
}

// 新增一個小元件用來儲存 ID
public class AnchorID : MonoBehaviour
{
    public int id;
}
