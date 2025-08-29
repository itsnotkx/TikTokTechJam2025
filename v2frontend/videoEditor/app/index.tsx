// import { Text, View } from "react-native";

// export default function Index() {
//   return (
//     <View
//       style={{
//         flex: 1,
//         justifyContent: "center",
//         alignItems: "center",
//       }}
//     >
//       <Text>Upload Video</Text>
//     </View>
//   );
// }


import React, { useState } from "react";
import { Button, View, Platform } from "react-native";
import { Video } from "expo-av";
import { router } from "expo-router";
import * as ImagePicker from "expo-image-picker";

export default function Index() {
  const [videoUri, setVideoUri] = useState<string | null>(null);

  const pickVideo = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
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
          // resizeMode="contain"
        />
      )}
    </View>
  );
}