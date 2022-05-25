using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MotionSensor : MonoBehaviour
{
    const float DEG2RAD = 0.01745329251f;
    ArticulationBody body;
    public Vector3 x { get { return transform.localPosition; } }
    public Vector3 x_t { get { return body.velocity; } }
    public Vector3 theta { get { return transform.rotation.eulerAngles * DEG2RAD; } }
    public Vector3 theta_t { get { return body.angularVelocity * DEG2RAD; } }
    void Awake()
    {
        body = GetComponent<ArticulationBody>();
    }
}
