import React from "react";
import { Button, View } from "react-native";
import { Video } from "expo-av";
import { router } from "expo-router";
import * as ImagePicker from "expo-image-picker";
import {useVideo} from "../context/VideoContext";

export default function Index() {
  // const [videoUri, setVideoUri] = useState<string | null>(null);
  const {videoUri, setVideoUri} = useVideo();

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

  return (
    <View style={{ justifyContent: "center", alignItems: "center", flex: 1}}>
      <Button title="Upload Video" onPress={pickVideo} />
      {videoUri && (
        <Video
          source={{ uri: videoUri }}
          style={{ width: "100%", height: 300 }}
          useNativeControls
        />
      )}
    </View>
  );
}