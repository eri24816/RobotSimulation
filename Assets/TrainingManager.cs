using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using WebSocketSharp;
using Newtonsoft.Json.Linq;
using System.Collections.Concurrent;



public class TrainingManager : MonoBehaviour
{
    public string wsServer = "ws://localhost:8765";
    WebSocket ws;
    readonly ConcurrentQueue<string> inMessages = new();
    public Robot robot;

    Vector3 prevPos = Vector3.zero;
    enum Phase
    {
        Freeze,
        Run
    }
    Phase phase;
    public float stepTime = 0.1f;
    public float currentStepTime = 0f;
    void Start()
    {
        ws = new WebSocket(wsServer);
        ws.OnMessage += (sender, e) => inMessages.Enqueue(e.Data);
        ws.Connect();
        EndStep();
        //StartStep();
    }
    private void Update()
    {
        while (inMessages.TryDequeue(out string m))
        {
            Recv(m);
        }
    }
    void FixedUpdate()
    {
        if(currentStepTime != 0f)
        {
            Debug.DrawLine(prevPos,robot.baseLink.x);
        }
        prevPos = robot.baseLink.x;
        currentStepTime += Time.fixedDeltaTime;
        if (phase == Phase.Run && currentStepTime > stepTime)
        {
            EndStep();
        }
    }
    void StartStep()
    {
        currentStepTime = 0;
        phase = Phase.Run;
        Time.timeScale = 1;
    }
    void EndStep()
    {
        phase = Phase.Freeze;
        Time.timeScale = 0;
        Send("state", robot.GetState());
    }
    void Send(string command, object data)
    {
        JObject jo = new();
        jo["command"] = command;
        jo["content"] = JObject.FromObject(data);
        ws.Send(jo.ToString());

        print($"send:\n {jo}");
    }
    void Recv(string message)
    {

        print($"recv:\n {message}");
        JObject jo = JObject.Parse(message);
        var content = jo["content"];
        var c = (string)jo["command"];

        switch (c)
        {
            case "action":
                {
                    robot.DoAction(content.ToObject<Robot.Action>());
                    StartStep();
                }
                break;
        }
    }

}
