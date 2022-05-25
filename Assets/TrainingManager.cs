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
    enum Phase
    {
        Freeze,
        Run
    }
    Phase phase;
    public float epochTime = 0.1f;
    public float currentEpochTime = 0f;
    void Start()
    {
        ws = new WebSocket(wsServer);
        ws.OnMessage += (sender, e) => inMessages.Enqueue(e.Data);
        ws.Connect();
        //StopEpoch();
        StartStep();
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

        currentEpochTime += Time.fixedDeltaTime;
        if (phase == Phase.Run && currentEpochTime > epochTime)
        {
            StopStep();
        }
    }
    void StartStep()
    {
        currentEpochTime = 0;
        phase = Phase.Run;
        Time.timeScale = 1;
    }
    void StopStep()
    {
        phase = Phase.Freeze;
        Time.timeScale = 0;
        Send("state", robot.GetState());
        print(robot.GetState());
    }
    void Send(string command, object data)
    {
        JObject jo = new();
        jo["command"] = command;
        jo["content"] = JObject.FromObject(data);
        ws.Send(jo.ToString());
        print(jo);
    }
    void Recv(string message)
    {
        print(message);
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
