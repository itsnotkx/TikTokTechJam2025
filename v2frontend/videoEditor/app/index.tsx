import React from "react";
import { Button, View, Text} from "react-native";
import { Video } from "expo-av";
import { router } from "expo-router";
import * as ImagePicker from "expo-image-picker";
import {useVideo} from "../context/VideoContext";
import { useState } from "react";

export default function Index() {
  // const [videoUri, setVideoUri] = useState<string | null>(null);
  const {videoUri, setVideoUri} = useVideo();
  const [resp, setResp] = useState<string | null>(null);

  const pickVideo = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      allowsEditing: true,
      quality: 1
    });

    if (!result.canceled) {
      console.log("result:", result.assets[0].uri);
      setVideoUri(result.assets[0].uri);
      router.push("/editor");
    }
  };

  const backendTest = async () => {
    fetch("http://localhost:8000/")
      .then((response) => response.json())
      .then((data) => setResp(data.message));

    console.log("resp:", resp);
  };

  return (
    <View style={{ justifyContent: "center", alignItems: "center", flex: 1}}>
      <Button title="Upload Video" onPress={pickVideo} />
      <Button title="Backend Testing" onPress={backendTest} />
      {resp ? <Text>{JSON.stringify(resp)}</Text> : <Text>No response</Text>}
      {/* {videoUri && (
        <Video
          source={{ uri: videoUri }}
          style={{ width: "100%", height: 300 }}
          useNativeControls
        />
      )} */}
    </View>
  );
}