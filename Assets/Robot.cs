using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class Robot : MonoBehaviour
{
    public struct State
    {
        public Vector2 baseLinkPos;
        public float baseLinkOrientation;
        public Vector2 baseLinkVelocity;
        public float baseLinkAngularVelocity;
        public List<float> wheelBaseOrientation;
        public List<float> wheelSpeed;
    }

    public struct Action
    {
        public List<float> voltage;
    }

    public MotionSensor baseLink;
    public MotionSensor[] wheelBase, wheel;

    List<Motor> motorList;

    // Start is called before the first frame update
    void Start()
    {
        baseLink = Util.GetOrAddComponent<MotionSensor>(transform, "base_link");
        wheelBase = new MotionSensor[]{
            Util.GetOrAddComponent<MotionSensor>(transform,"left_back_wheel_base"),
            Util.GetOrAddComponent<MotionSensor>(transform,"left_front_wheel_base"),
            Util.GetOrAddComponent<MotionSensor>(transform,"right_back_wheel_base"),
            Util.GetOrAddComponent<MotionSensor>(transform,"right_front_wheel_base")
        };
        wheel = new MotionSensor[]{
            Util.GetOrAddComponent<MotionSensor>(transform,"left_back_wheel"),
            Util.GetOrAddComponent<MotionSensor>(transform,"left_front_wheel"),
            Util.GetOrAddComponent<MotionSensor>(transform,"right_back_wheel"),
            Util.GetOrAddComponent<MotionSensor>(transform,"right_front_wheel")
        };
        motorList = new List<Motor>
        {
            Util.GetOrAddComponent<Motor>(transform,"left_back_wheel_base"),
            Util.GetOrAddComponent<Motor>(transform,"left_front_wheel_base"),
            Util.GetOrAddComponent<Motor>(transform,"right_back_wheel_base"),
            Util.GetOrAddComponent<Motor>(transform,"right_front_wheel_base"),
            Util.GetOrAddComponent<Motor>(transform,"left_back_wheel"),
            Util.GetOrAddComponent<Motor>(transform,"left_front_wheel"),
            Util.GetOrAddComponent<Motor>(transform,"right_back_wheel"),
            Util.GetOrAddComponent<Motor>(transform,"right_front_wheel")
        };

        DoAction(new Action
        {
            voltage = new List<float>{200,200,200,200,200,200,200,200 }
        }
        );
    }


    // Update is called once per frame
    void Update()
    {
        //print(JsonUtility.ToJson(GetState()));
    }
    public State GetState()
    {
        State state = new()
        {
            baseLinkPos = baseLink.x,
            baseLinkOrientation = baseLink.theta.y,
            baseLinkVelocity = baseLink.x_t,
            baseLinkAngularVelocity = baseLink.theta_t.y
        };
        state.wheelBaseOrientation = new();
        foreach (MotionSensor w in wheelBase) state.wheelBaseOrientation.Add(w.theta.y);
        state.wheelSpeed = new();
        foreach (MotionSensor w in wheel) state.wheelSpeed.Add(w.theta_t.x);
        return state;
    }
    public void DoAction(Action action)
    {
        
        for(int i = 0; i < 8; i++)
        {
            motorList[i].SetVoltage(action.voltage[i]);
        }
    }
}
