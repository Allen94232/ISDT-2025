using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class Lab4_GameManager : MonoBehaviour
{
    // add instance
    public static Lab4_GameManager instance;
    private void Awake()
    {
        if (instance == null)
        {
            instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    public bool isGaming = false;
    public bool isExchanging = false;
    public bool isGameOver = false; // 新增遊戲結束狀態

    public GameObject gemGenerationArea;

    public GameObject gemPrefab_1;
    public GameObject gemPrefab_2;
    public GameObject gemPrefab_3;
    public GameObject gemPrefab_4;
    public GameObject gemPrefab_5;

    // Game UI
    public GameObject gameUI; // 遊戲 UI 面板

    // 寶石管理
    private List<GameObject> activeGems = new List<GameObject>();
    private List<GameObject> gemPrefabs = new List<GameObject>();
    private const int INITIAL_MAX_GEMS = 5; // 初始最大寶石數量
    private int currentMaxGems = 5; // 當前最大寶石數量（動態調整）
    
    // 生成區域的範圍
    private Bounds generationBounds;
    private bool wasExchanging = false;

    private int exchangeCount = 0;
    private const int MAX_EXCHANGES = 3; // 最大交換次數

    private void Start()
    {
        isGaming = false;
        isExchanging = false;
        isGameOver = false;
        currentMaxGems = INITIAL_MAX_GEMS; // 初始化當前最大值
        
        // 初始化寶石預製件列表
        InitializeGemPrefabs();
        
        // 獲取生成區域的邊界
        if (gemGenerationArea != null)
        {
            // 首先嘗試從 Renderer 獲取邊界
            Renderer areaRenderer = gemGenerationArea.GetComponent<Renderer>();
            if (areaRenderer != null)
            {
                generationBounds = areaRenderer.bounds;
                Debug.Log($"Using Renderer bounds: {generationBounds}");
            }
            else
            {
                // 嘗試從 Collider 獲取邊界
                Collider areaCollider = gemGenerationArea.GetComponent<Collider>();
                if (areaCollider != null)
                {
                    generationBounds = areaCollider.bounds;
                    Debug.Log($"Using Collider bounds: {generationBounds}");
                }
                else
                {
                    // 如果沒有 Renderer 或 Collider，使用 Transform 的位置作為中心點
                    generationBounds = new Bounds(gemGenerationArea.transform.position, Vector3.one * 10f);
                    Debug.LogWarning($"No Renderer or Collider found on gemGenerationArea. Using default bounds around Transform position: {generationBounds}");
                }
            }
        }
        else
        {
            Debug.LogError("gemGenerationArea is null! Please assign it in the inspector.");
        }
    }
    private void InitializeGemPrefabs()
    {
        gemPrefabs.Clear();
        
        if (gemPrefab_1 != null) gemPrefabs.Add(gemPrefab_1);
        if (gemPrefab_2 != null) gemPrefabs.Add(gemPrefab_2);
        if (gemPrefab_3 != null) gemPrefabs.Add(gemPrefab_3);
        if (gemPrefab_4 != null) gemPrefabs.Add(gemPrefab_4);
        if (gemPrefab_5 != null) gemPrefabs.Add(gemPrefab_5);
        
        if (gemPrefabs.Count == 0)
        {
            Debug.LogWarning("No gem prefabs assigned to Lab4_GameManager!");
        }
    }

    private void Update()
    {
        // 如果遊戲結束，不執行其他更新邏輯
        if (isGameOver)
        {
            return;
        }

        if (!isGaming)
        {
            return;
        }

        // 檢查 isExchanging 狀態變化
        if (wasExchanging != isExchanging)
        {
            if (isExchanging)
            {
                // 變成 exchanging，移除所有寶石並重置最大值
                RemoveAllGems();
                ResetMaxGems();
            }
            else
            {
                // 變成 not exchanging，開始生成寶石
                GenerateGemsToMax();
            }
            wasExchanging = isExchanging;
        }

        if (!isExchanging)
        {
            // 清理已被銷毀的寶石引用
            CleanupDestroyedGems();
            
            // 如果寶石數量不足，補充到當前最大值
            if (activeGems.Count < currentMaxGems)
            {
                int gemsToGenerate = currentMaxGems - activeGems.Count;
                for (int i = 0; i < gemsToGenerate; i++)
                {
                    GenerateRandomGem();
                }
            }
        }
    }

    private void CleanupDestroyedGems()
    {
        // 移除已被銷毀的寶石引用（從後往前遍歷避免索引問題）
        for (int i = activeGems.Count - 1; i >= 0; i--)
        {
            if (activeGems[i] == null)
            {
                activeGems.RemoveAt(i);
            }
        }
    }

    private void GenerateGemsToMax()
    {
        // 生成寶石直到達到當前最大數量
        for (int i = activeGems.Count; i < currentMaxGems; i++)
        {
            GenerateRandomGem();
        }
    }

    private void GenerateRandomGem()
    {
        if (gemPrefabs.Count == 0)
        {
            Debug.LogWarning("Cannot generate gem: No gem prefabs available!");
            return;
        }

        // 每次生成寶石時重新計算生成區域的邊界
        UpdateGenerationBounds();

        // 隨機選擇一個寶石預製件
        GameObject randomGemPrefab = gemPrefabs[Random.Range(0, gemPrefabs.Count)];
        
        // 在生成區域內隨機位置生成
        Vector3 randomPosition = GetRandomPositionInArea();
        
        // 隨機旋轉
        Quaternion randomRotation = Quaternion.Euler(
            Random.Range(0f, 360f),
            Random.Range(0f, 360f),
            Random.Range(0f, 360f)
        );
        
        // 生成寶石
        GameObject newGem = Instantiate(randomGemPrefab, randomPosition, randomRotation);
        
        // 添加到活躍寶石列表
        activeGems.Add(newGem);
        
        Debug.Log($"Generated gem at position: {randomPosition}. Current gems: {activeGems.Count}/{currentMaxGems}");
    }

    // 新增：更新生成區域邊界的方法
    private void UpdateGenerationBounds()
    {
        if (gemGenerationArea != null)
        {
            // 首先嘗試從 Renderer 獲取邊界
            Renderer areaRenderer = gemGenerationArea.GetComponent<Renderer>();
            if (areaRenderer != null)
            {
                generationBounds = areaRenderer.bounds;
            }
            else
            {
                // 嘗試從 Collider 獲取邊界
                Collider areaCollider = gemGenerationArea.GetComponent<Collider>();
                if (areaCollider != null)
                {
                    generationBounds = areaCollider.bounds;
                }
                else
                {
                    // 如果沒有 Renderer 或 Collider，使用 Transform 的位置作為中心點
                    generationBounds = new Bounds(gemGenerationArea.transform.position, Vector3.one * 10f);
                }
            }
        }
    }

    private Vector3 GetRandomPositionInArea()
    {
        // 在生成區域的邊界內隨機生成位置
        Vector3 randomPosition = new Vector3(
            Random.Range(generationBounds.min.x, generationBounds.max.x),
            0.5f,
            Random.Range(generationBounds.min.z, generationBounds.max.z)
        );
        
        Debug.Log($"Generated position: {randomPosition}, Bounds: min={generationBounds.min}, max={generationBounds.max}");
        return randomPosition;
    }

    private void RemoveAllGems()
    {
        // 銷毀所有活躍的寶石
        foreach (GameObject gem in activeGems)
        {
            if (gem != null)
            {
                Destroy(gem);
            }
        }
        
        // 清空列表
        activeGems.Clear();
        
        Debug.Log("All gems removed");
    }

    private void ResetMaxGems()
    {
        // 重置最大寶石數量為初始值
        currentMaxGems = INITIAL_MAX_GEMS;
        Debug.Log($"Max gems reset to {currentMaxGems}");
    }

    public void ToggleExchanging(bool newExchangingState)
    {
        if (this.isExchanging == newExchangingState)
        {
            return;
        }

        if (!newExchangingState)
        {
            exchangeCount++;
            Debug.Log($"Exchange count: {exchangeCount}");
        }

        if (exchangeCount >= MAX_EXCHANGES)
        {
            // 觸發遊戲結束
            GameOver();
            return;
        }

        if (newExchangingState && gameUI != null)
        {
            gameUI.transform.GetChild(0).gameObject.GetComponent<TextMeshProUGUI>().text = "Put minecart here";
            gameUI.SetActive(true);
        }
        else if (!newExchangingState && gameUI != null)
        {
            gameUI.SetActive(false);
        }

        this.isExchanging = newExchangingState;
        Debug.Log($"Exchanging state changed to: {newExchangingState}");
    }

    public void GameOver()
    {
        isGameOver = true;
        isGaming = false;
        isExchanging = false;

        // 停止所有寶石生成
        RemoveAllGems();

        // 顯示 Game Over UI
        if (gameUI != null)
        {
            gameUI.transform.GetChild(0).gameObject.GetComponent<TextMeshProUGUI>().text = "Game Over! Maximum exchanges reached.";

            // set gameUI parent to root
            gameUI.transform.SetParent(null);

            gameUI.SetActive(true);
        }

        Debug.Log("Game Over! Maximum exchanges reached.");
    }

    public void RestartGame4()
    {
        // 重置遊戲狀態
        isGameOver = false;
        isGaming = false;
        isExchanging = false;
        exchangeCount = 0;
        wasExchanging = false;

        // 清理所有寶石
        RemoveAllGems();

        // 重置最大寶石數量
        ResetMaxGems();

        // 隱藏 Game Over UI
        if (gameUI != null)
        {
            gameUI.transform.GetChild(0).gameObject.GetComponent<TextMeshProUGUI>().text = "Put minecart here";
            gameUI.SetActive(false);
        }

        Debug.Log("Game restarted!");
    }

    // 公開方法供外部調用
    public int GetActiveGemCount()
    {
        CleanupDestroyedGems();
        return activeGems.Count;
    }

    public List<GameObject> GetActiveGems()
    {
        CleanupDestroyedGems();
        return new List<GameObject>(activeGems);
    }

    public int GetExchangeCount()
    {
        return exchangeCount;
    }

    public int GetMaxExchanges()
    {
        return MAX_EXCHANGES;
    }

    public int GetCurrentMaxGems()
    {
        return currentMaxGems;
    }

    // 寶石碰到礦車時調用：增加最大值並生成新寶石
    public void CollectGem(GameObject gem)
    {
        if (activeGems.Contains(gem))
        {
            // 不從列表中移除，也不銷毀寶石
            // 只增加最大值
            currentMaxGems++;
            Debug.Log($"Gem collected! Max gems increased to {currentMaxGems}");
            
            // 寶石會保留在場上，系統會自動補充一個新寶石到新的最大值
        }
    }

    // 清理方法，在場景切換或遊戲結束時調用
    private void OnDestroy()
    {
        RemoveAllGems();
    }
}
