#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <QMC5883L.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <Adafruit_BMP085.h>
#include <esp_now.h>
#include <WiFi.h>
#include <TinyGPS++.h>

uint8_t broadcastAddress[] = {0xFC, 0xB4, 0x67, 0x78, 0x8D, 0xC8};


Adafruit_MPU6050 mpu;
Adafruit_BMP085 bmp;
QMC5883L compass;
// The TinyGPS++ object
TinyGPSPlus gps;

int Buzzer = 4; //for ESP32
float height = 46; // Reference height to start buzzer

//struct for Data
typedef struct struct_message {
  float T_BMP;
  float Pressure;
  float Altitude;
  float SeaLevelPressure;
  float RealAltitude;
  float accelerationX;
  float accelerationY;
  float accelerationZ;
  float gyroX;
  float gyroY;
  float gyroZ;
  float T_MPU;
  float Compass;
  float Latitude;
  float Longitude;
  float Altitude_GPS;
  float DateMonth;
  float DateDay;
  float DateYear;
  float SatValue;
} struct_message;
struct_message myData;

esp_now_peer_info_t peerInfo;

// callback when data is sent
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  Serial.print("\r\nLast Packet Send Status:\t");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
}
 

void readFile(fs::FS &fs, const char * path){
  Serial.printf("Reading file: %s\n", path);

  File file = fs.open(path);
  if(!file){
    Serial.println("Failed to open file for reading");
    return;
  }

  Serial.print("Read from file: ");
  while(file.available()){
    Serial.write(file.read());
  }
  file.close();
}

void writeFile(fs::FS &fs, const char * path, const char * message){
  Serial.printf("Writing file: %s\n", path);

  File file = fs.open(path, FILE_WRITE);
  if(!file){
    Serial.println("Failed to open file for writing");
    return;
  }
  if(file.print(message)){
    Serial.println("File written");
  } else {
    Serial.println("Write failed");
  }
  file.close();
}

void appendFile(fs::FS &fs, const char * path, String message){
  Serial.printf("Appending to file: %s\n", path);

  File file = fs.open(path, FILE_APPEND);
  if(!file){
    Serial.println("Failed to open file for appending");
    return;
  }
  if(file.print(message)){
      Serial.println("Message appended");
  } else {
    Serial.println("Append failed");
  }
  file.close();
}

void deleteFile(fs::FS &fs, const char * path){
  Serial.printf("Deleting file: %s\n", path);
  if(fs.remove(path)){
    Serial.println("File deleted");
  } else {
    Serial.println("Delete failed");
  }
}

#define RXD2 16
#define TXD2 17
void displayInfo()
{
  Serial.print(F("Location: ")); 
  if (gps.location.isValid())
  {
    Serial.print(gps.location.lat(), 6);
    Serial.print(F(", "));
    Serial.print(gps.location.lng(), 6);
  }
  else
  {
    Serial.print(F("INVALID"));
  }
  Serial.print(F("  Altitude: ")); 
  if (gps.altitude.isValid())
  {
    Serial.print(gps.altitude.meters());
    Serial.print(F(" m "));
  }
  else
  {
    Serial.print(F("INVALID"));
  }

  Serial.print(F("  Date/Time: "));
  if (gps.date.isValid())
  {
    Serial.print(gps.date.month());
    Serial.print(F("/"));
    Serial.print(gps.date.day());
    Serial.print(F("/"));
    Serial.print(gps.date.year());
  }
  else
  {
    Serial.print(F("INVALID"));
  }

  Serial.print(F(" "));
  if (gps.time.isValid())
  {
    if (gps.time.hour() < 10) Serial.print(F("0"));
    Serial.print(gps.time.hour());
    Serial.print(F(":"));
    if (gps.time.minute() < 10) Serial.print(F("0"));
    Serial.print(gps.time.minute());
    Serial.print(F(":"));
    if (gps.time.second() < 10) Serial.print(F("0"));
    Serial.print(gps.time.second());
    Serial.print(F("."));
    if (gps.time.centisecond() < 10) Serial.print(F("0"));
    Serial.print(gps.time.centisecond());
  }
  else
  {
    Serial.print(F("INVALID"));
  }

  Serial.print(F("  Satellite: ")); 
  if (gps.satellites.isValid())
  {
    Serial.print(gps.satellites.value());
  }
  else
  {
    Serial.print(F("INVALID"));
  }
  Serial.println();
}


void setup(){

  pinMode (Buzzer, OUTPUT);
  Serial2.begin(9600, SERIAL_8N1,RXD2,TXD2);
  Serial.begin(115200); // One Serial.begin is enough to call setup function - will be tested.
  delay(500);

  Serial.println(F("DeviceExample.ino"));
  Serial.println(F("A simple demonstration of TinyGPS++ with an attached GPS module"));
  Serial.print(F("Testing TinyGPS++ library v. ")); Serial.println(TinyGPSPlus::libraryVersion());
  Serial.println();

  Wire.begin();

	compass.init();
	compass.setSamplingRate(50);

	// Serial.begin(9600);
	Serial.println("QMC5883L Compass Demo");
	Serial.println("Turn compass in all directions to calibrate....");


  // Serial.begin(9600);

   WiFi.mode(WIFI_STA);

  // Init ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  // Once ESPNow is successfully Init, we will register for Send CB to
  // get the status of Trasnmitted packet
  esp_now_register_send_cb(OnDataSent);
  
  // Register peer
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  
  // Add peer        
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to add peer");
    return;
  }

 if (!bmp.begin()) {
	Serial.println("Could not find a valid BMP085/BMP180 sensor, check wiring!");
	while (1) {}
  }

  while (!Serial)
    delay(10); // will pause Zero, Leonardo, etc until serial console opens

  Serial.println("Adafruit MPU6050 test!");

  // Try to initialize!
  if (!mpu.begin())  {
    Serial.println("Failed to find MPU6050 chip");
    Serial.println("Could not find a valid BMP085/BMP180 sensor, check wiring!");
    while (1) {
      delay(10); //burası silinebilir
    }

  }
  
  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.print("Accelerometer range set to: ");
  switch (mpu.getAccelerometerRange()) {
  case MPU6050_RANGE_2_G:
    Serial.println("+-2G");
    break;
  case MPU6050_RANGE_4_G:
    Serial.println("+-4G");
    break;
  case MPU6050_RANGE_8_G:
    Serial.println("+-8G");
    break;
  case MPU6050_RANGE_16_G:
    Serial.println("+-16G");
    break;
  }
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  Serial.print("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
  case MPU6050_RANGE_250_DEG:
    Serial.println("+- 250 deg/s");
    break;
  case MPU6050_RANGE_500_DEG:
    Serial.println("+- 500 deg/s");
    break;
  case MPU6050_RANGE_1000_DEG:
    Serial.println("+- 1000 deg/s");
    break;
  case MPU6050_RANGE_2000_DEG:
    Serial.println("+- 2000 deg/s");
    break;
  }

  mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);
  Serial.print("Filter bandwidth set to: ");
  switch (mpu.getFilterBandwidth()) {
  case MPU6050_BAND_260_HZ:
    Serial.println("260 Hz");
    break;
  case MPU6050_BAND_184_HZ:
    Serial.println("184 Hz");
    break;
  case MPU6050_BAND_94_HZ:
    Serial.println("94 Hz");
    break;
  case MPU6050_BAND_44_HZ:
    Serial.println("44 Hz");
    break;
  case MPU6050_BAND_21_HZ:
    Serial.println("21 Hz");
    break;
  case MPU6050_BAND_10_HZ:
    Serial.println("10 Hz");
    break;
  case MPU6050_BAND_5_HZ:
    Serial.println("5 Hz");
    break;
  }

  Serial.println("");
  delay(100);


  // Serial.begin(115200);
  if(!SD.begin(5)){
    Serial.println("Card Mount Failed");
    return;
  }
  uint8_t cardType = SD.cardType();

  if(cardType == CARD_NONE){
    Serial.println("No SD card attached");
    return;
  }

  Serial.print("SD Card Type: ");
  if(cardType == CARD_MMC){
    Serial.println("MMC");
  } else if(cardType == CARD_SD){
    Serial.println("SDSC");
  } else if(cardType == CARD_SDHC){
    Serial.println("SDHC");
  } else {
    Serial.println("UNKNOWN");
  }

  uint64_t cardSize = SD.cardSize() / (1024 * 1024);
  Serial.printf("SD Card Size: %lluMB\n", cardSize);

  writeFile(SD, "/data.txt", "Sensor Readings \n");
  // appendFile(SD, "/data.txt", "World!\n");
  //readFile(SD, "/data.txt");
  Serial.printf("Total space: %lluMB\n", SD.totalBytes() / (1024 * 1024));
  Serial.printf("Used space: %lluMB\n", SD.usedBytes() / (1024 * 1024));
}

void loop(){
  // This sketch displays information every time a new sentence is correctly encoded.
  while (Serial2.available() > 0)
    if (gps.encode(Serial2.read()))
      displayInfo();
      delay(1000);

  if (millis() > 5000 && gps.charsProcessed() < 10)
  {
    Serial.println(F("No GPS detected: check wiring."));
    while(true);
  }

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  delay(500);

  int heading = compass.readHeading();
	if(heading==0) {
		/* Still calibrating, so measure but don't print */
	} else {
    // Measure but do not print any output in Serial Monitor in this code, only send the receiver and write to SD card.
	}


    myData.T_BMP=bmp.readTemperature();
    myData.Pressure=bmp.readPressure();
    myData.Altitude=bmp.readAltitude();
    myData.SeaLevelPressure=bmp.readSealevelPressure();
    myData.RealAltitude=bmp.readAltitude(102000);
    myData.accelerationX=a.acceleration.x;
    myData.accelerationY=a.acceleration.y;
    myData.accelerationZ=a.acceleration.z;
    myData.gyroX=g.gyro.x;
    myData.gyroY=g.gyro.y;
    myData.gyroZ=g.gyro.z;
    myData.T_MPU=temp.temperature;
    myData.Compass=compass.readHeading();
    myData.Latitude=gps.location.lat();
    myData.Longitude=gps.location.lng();
    myData.Altitude_GPS=gps.altitude.meters();
    myData.DateMonth=gps.date.month();
    myData.DateDay=gps.date.day();
    myData.DateYear=gps.date.year();
    myData.SatValue=gps.satellites.value();
    esp_err_t result = esp_now_send(broadcastAddress, (uint8_t *) &myData, sizeof(myData));
   
  if (result == ESP_OK) {
    Serial.println("Sent with success");
  }
  else {
    Serial.println("Error sending the data");
  }

    Serial.println();
    delay(200);
    String bmp_text = "Temperature_BMP: " + String(myData.T_BMP) + " , Pressure: " + String(myData.Pressure) + " , Altitude: " + String(myData.Altitude) + " , Sea Level Pressure: " + String(myData.SeaLevelPressure) + " , Real Altitude: "+ String(myData.RealAltitude);
    String mpu_text = " , Acceleration_x: "+ String(myData.accelerationX) + " , Acceleration_y: " +String(myData.accelerationY) + ", Acceleration_z: " + String(myData.accelerationZ) + " , Gyro_x: " + String(myData.gyroX) +  " , Gyro_y: " + String(myData.gyroY) + " , Gyro_z: " + String(myData.gyroZ) + " , Temperature_MPU: " + String(myData.T_MPU);
    String hmc_text = " , Magnetometer: " + String(myData.Compass) + " , ";
    String gps_text = " , Latitude: " + String(myData.Latitude) + " , " + " , Longitude: " + String(myData.Longitude) + " , Altitude_GPS: " + String(myData.Altitude_GPS) + " , Month: " + String(myData.DateMonth) + " , Day: " + String(myData.DateDay) + " , Year: " + String(myData.DateYear) + " , Satellite: " + String(myData.SatValue) + " , ";
    appendFile(SD, "/data.txt",bmp_text);
    appendFile(SD, "/data.txt",mpu_text);
    appendFile(SD, "/data.txt",hmc_text);
    appendFile(SD, "/data.txt",gps_text);
    Serial.println("");
    //readFile(SD, "/data.txt");

    // Check if the altitude is less than reference altitude height to indicate landing

    // if (height > myData.Altitude) { 
    if (height > myData.Altitude) {
    // Trigger the buzzer
    tone(Buzzer, 400); // Start buzzer with frequency 400Hz
    delay(1000); // Keep buzzer on for 1 second
    noTone(Buzzer); // Stop the buzzer
    }
}
