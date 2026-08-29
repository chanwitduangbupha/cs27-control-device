#define LED_PIN 12

String command = "";

void setup() {
  Serial.begin(9600);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.println("Ready");
}

void loop() {

  if (Serial.available() > 0) {

    command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "RED") {
      blinkLED(1);
      Serial.println("Red - Blink 1");
    }

    else if (command == "GREEN") {
      blinkLED(2);
      Serial.println("Green - Blink 2");
    }

    else if (command == "BLUE") {
      blinkLED(3);
      Serial.println("Blue - Blink 3");
    }

    else if (command == "OFF") {
      digitalWrite(LED_PIN, LOW);
      Serial.println("LED OFF");
    }

    else {
      Serial.println("Unknown Command");
    }
  }
}

void blinkLED(int times) {

  for (int i = 0; i < times; i++) {

    digitalWrite(LED_PIN, HIGH);
    delay(300);

    digitalWrite(LED_PIN, LOW);
    delay(300);
  }
}