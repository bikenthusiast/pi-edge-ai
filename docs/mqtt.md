# MQTT interface

The pipeline publishes to an MQTT broker so that other devices — first of all
the MagicMirror — can react to detections without touching `events.db`.
SQLite remains the source of truth; MQTT is a best-effort notification
channel. The reasoning is in [ADR 0003](adr/0003-mqtt-als-event-schnittstelle.md).

## Topics

All topics live under `edge/<device>/`, where `<device>` defaults to the Pi's
hostname (`--device` overrides it).

| Topic | Retained | Sent when | Payload |
|---|---|---|---|
| `events` | no | an episode closes | one detection, same data as the SQLite row |
| `presence` | yes | an episode opens or closes | what is in view right now |
| `status` | yes | connect, clean shutdown, or crash (Last Will) | `{"online": …}` |

`presence` is retained so that a mirror booting later still knows the current
state. After a crash it can be stale — consumers must treat it as valid only
while `status.online` is `true`.

## Payloads (schema v1)

Every payload is JSON with a `"v"` field. Consumers must ignore versions they do
not know rather than guess.

```json
// edge/<device>/events
{"v":1,"id":42,"ts":"2026-09-19T08:30:00.000+00:00","label":"tabby",
 "score":0.93,"source":"camera0","duration_ms":4200.0,
 "model":"mobilenet_v2_1.0_224_quant.tflite"}

// edge/<device>/presence
{"v":1,"ts":"2026-09-19T08:29:56.000+00:00","source":"camera0",
 "present":true,"label":"tabby"}

// edge/<device>/status
{"v":1,"online":true}
```

`id` and `ts` match the row in `events.db`, so a consumer can always fetch
more detail from the log. `ts` of an event is when the episode *closed*; it
started `duration_ms` earlier. `score` is the peak score of the episode.

## Broker setup on the Pi

`scripts/bootstrap_pi.sh` installs Mosquitto. Mosquitto 2.x only listens on
localhost by default, so the LAN side needs a config. Two listeners keep the
local pipeline simple and the network side locked down:

```conf
# /etc/mosquitto/conf.d/edge.conf
per_listener_settings true

# Loopback: the pipeline on this Pi, no credentials.
listener 1883 127.0.0.1
allow_anonymous true

# LAN: the mirror and other consumers, password + ACL required.
listener 1884
allow_anonymous false
password_file /etc/mosquitto/passwd
acl_file /etc/mosquitto/acl
```

```conf
# /etc/mosquitto/acl — the mirror may read, never write
user mirror
topic read edge/#
```

```bash
sudo mosquitto_passwd -c /etc/mosquitto/passwd mirror    # prompts for the password
sudo chown root:mosquitto /etc/mosquitto/passwd /etc/mosquitto/acl
sudo chmod 640 /etc/mosquitto/passwd /etc/mosquitto/acl  # broker runs as 'mosquitto'
sudo systemctl restart mosquitto
```

These files stay on the Pi and are never committed.

## Running and watching

```bash
make pipeline-mqtt     # 60 s run on the Pi, publishing to localhost
make mqtt-watch        # follow edge/# on the Pi
```

If the pipeline connects to a broker that requires credentials, pass them via
`MQTT_USERNAME` / `MQTT_PASSWORD` in the environment — not on the command line.

## Failure behaviour

| Situation | What consumers see |
|---|---|
| Broker down at start | pipeline runs normally; paho reconnects in the background |
| Broker down for a while | up to 1000 messages queued in RAM, further ones rejected; SQLite has all |
| Pipeline stopped cleanly | `status` → offline, `presence` → not present |
| Pipeline crashed | Last Will sets `status` → offline immediately |
| Pi loses power / network | Last Will after at most 45 s (1.5 × keepalive) |
