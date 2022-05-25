using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Motor : MonoBehaviour
{
    ArticulationBody body;
    void Awake()
    {
        body = GetComponent<ArticulationBody>();
    }
    public void SetVoltage(float voltage)
    {
        var drive = body.xDrive;
        drive.targetVelocity = voltage;
        body.xDrive = drive;
    }
}
