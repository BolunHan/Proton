# Server Setup

same as the env setup for local training

# Relay Setup

We use an Aliyun Server (with public IPv4) as relay.

A ECS instance is set up at `47.94.191.124` (2vCPU, 1GiB Mem).

The ssh credentials are:

```bash
sshpass -p TZnqE6oBWs9^sT2^ ssh root@47.94.191.124
```

For security reason, password login is disabled `PasswordAuthentication no`. Contact [Bolun](mailto:Bolun.Han@outlook.com) for login support.

The relay have a WireGuard service, providing Basic VPN tunneling.

```md
Address: 47.94.191.124
Port: 59873
PrivateKey: CHK4HrXjc+rRDL84IJtsUSmaD7f/5aZ6CN5WdQdts1I=
PublicKey: l3NSPRHISlWgCLsPqjkgK5zyUdxiXv2cEQ62EZxBkgI=
IPv4: 10.8.0.1/24
IPv6: fd0d:86fa:c3bc::1/64
```

The training server uses following config

```md
PrivateKey: 2JEyET/hVG7rI4C495tnu/L2VHwr1LRx0gK+2WVABW0=
PublicKey: eF6gILmGwkdA7nh06CpKaX/2gMHiq9w6UL2S+ijP5AM=
```

The config file `/etc/wireguard/wg0.conf` is

```
[Interface]
PrivateKey = 2JEyET/hVG7rI4C495tnu/L2VHwr1LRx0gK+2WVABW0=
Address = 10.8.0.2/24
Address = fd0d:86fa:c3bc::2/64

[Peer]
PublicKey = l3NSPRHISlWgCLsPqjkgK5zyUdxiXv2cEQ62EZxBkgI=
AllowedIPs = 10.8.0.0/24, fd0d:86fa:c3bc::/64
Endpoint = 47.94.191.124:59873
```

# WireGuard Client Setup

I have added this config to the relay server. Note that only one ip (connection) is allowed at on time.

The key value pair

```md
PrivateKey: ICBpb+x+3IoVu8FVpcKJPjIlaM14QiCUN/xU6+OyHkQ=
PublicKey: NCsInm9SMGwcwJnuNNzG4wYByUG9DNxYpac/T7Mp2C8=
```

The config file

```
[Interface]
PrivateKey = ICBpb+x+3IoVu8FVpcKJPjIlaM14QiCUN/xU6+OyHkQ=
Address = 10.8.0.3/24
Address = fd0d:86fa:c3bc::3/64

[Peer]
PublicKey = l3NSPRHISlWgCLsPqjkgK5zyUdxiXv2cEQ62EZxBkgI=
AllowedIPs = 10.8.0.0/24, fd0d:86fa:c3bc::/64
Endpoint = 47.94.191.124:59873
```

If you need more connection, contact [Bolun](mailto:Bolun.Han@outlook.com)

# Training Server Access

After you connect the WireGuard VPN, you can task the server using SSH or RDP.

For SSH Access

```bash
sshpass -p 348995 ssh bolun@10.8.0.2
```

For RDP access: use any remote desktop client to connect to `10.8.0.2`, with same user `bolun` and pwd `348995`.

# Add More Peer for WireGuard

See the docs at https://www.digitalocean.com/community/tutorials/how-to-set-up-wireguard-on-ubuntu-20-04#step-7-configuring-a-wireguard-peer

1. install WireGuard with 
    ```bash
    sudo apt update
    sudo apt install wireguard
    ```
2. Generate key pair
    ```
    wg genkey | sudo tee /etc/wireguard/private.key
    sudo chmod go= /etc/wireguard/private.key
    sudo cat /etc/wireguard/private.key | wg pubkey | sudo tee /etc/wireguard/public.key
    ```
3. Add PubKey to WireGuard Server
    run this command at server `47.94.191.124`
    ```
    sudo wg set wg0 peer <the_pub_key_base64> allowed-ips 10.8.0.4,fd0d:86fa:c3bc::4
    ```
4. Add Config File at Local
    ```
    [Interface]
    PrivateKey = <the_private_key_base64>
    Address = 10.8.0.4/24
    Address = fd0d:86fa:c3bc::4/64
    
    [Peer]
    PublicKey = l3NSPRHISlWgCLsPqjkgK5zyUdxiXv2cEQ62EZxBkgI=
    AllowedIPs = 10.8.0.0/24, fd0d:86fa:c3bc::/64
    Endpoint = 47.94.191.124:59873
    ```
5. Connect the Tunneling Service
    ```
    sudo wg-quick up <the_config_file_name>
    ```
