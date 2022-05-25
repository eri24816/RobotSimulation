using UnityEngine;

class Util
{

    public static T GetOrAddComponent<T>(Transform transform,string childName) where T : Component
    {
        var res = RecursiveFindChild(transform, childName).GetComponent<T>();
        if (res != null)
            return res;
        return RecursiveFindChild(transform, childName).gameObject.AddComponent<T>();
    }
    public static Transform RecursiveFindChild(Transform parent, string childName)
    {
        foreach (Transform child in parent)
        {
            if (child.name == childName)
            {
                return child;
            }
            else
            {
                Transform found = RecursiveFindChild(child, childName);
                if (found != null)
                {
                    return found;
                }
            }
        }
        return null;
    }

}
