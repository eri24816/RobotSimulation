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
    [SerializeField]
    GameObject target;
    [SerializeField]
    float logProb = 0.01f;
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

        if (Random.Range(0, 1f) < logProb)
            print($"send:\n {jo}");
    }
    void Recv(string message)
    {
        if(Random.Range(0, 1f)<logProb)
            print($"recv:\n {message}");
        JObject jo = JObject.Parse(message);
        var content = jo["content"];
        var c = (string)jo["command"];

        switch (c)
        {
            case "action":
                    robot.DoAction(content.ToObject<Robot.Action>());
                    StartStep();
                break;
            case "new target":
                Vector3 pos = content["pos"].ToObject<Vector3>();
                target.transform.position = pos;
                Transform baselink = robot.transform.Find("base_link");
                baselink.GetComponent<ArticulationBody>().enabled = false;
                baselink.rotation = Quaternion.identity;
                //baselink.transform.Rotate(0, 0.2f, 0);
                baselink.GetComponent<ArticulationBody>().enabled = true;
                break;
        }
    }

}
