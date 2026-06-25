#define RX2_PIN 16
#define TX2_PIN 17

void setup() {
  Serial.begin(115200);  // USB to Python
  Serial2.begin(9600, SERIAL_8N1, RX2_PIN, TX2_PIN); // previous/next chain
}

void loop() {
  // Real packet from previous node -> send to Python
  if (Serial2.available()) {
    String packetFromPrevious = Serial2.readStringUntil('\n');
    packetFromPrevious.trim();

    if (packetFromPrevious.length() > 0) {
      Serial.println(packetFromPrevious);
    }
  }

  // Modified packet from Python -> send onwards
  if (Serial.available()) {
    String packetFromPython = Serial.readStringUntil('\n');
    packetFromPython.trim();

    if (packetFromPython.length() > 0) {
      Serial2.println(packetFromPython);
    }
  }
}