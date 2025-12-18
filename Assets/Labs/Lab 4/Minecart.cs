using UnityEngine;

public class Minecart : MonoBehaviour
{
    private void OnTriggerEnter(Collider other)
    {
        if(other.CompareTag("Gem"))
        {
            // 改為調用 CollectGem 而不是 RemoveGem
            // CollectGem 會增加最大值但不刪除寶石
            Lab4_GameManager.instance.CollectGem(other.gameObject);
        }
    }
}
