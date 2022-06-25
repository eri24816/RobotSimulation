using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Target : MonoBehaviour
{
    Vector3 ppos;
    public bool moved = false;
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        moved = (ppos != transform.position);
        ppos = transform.position;
    }
}
