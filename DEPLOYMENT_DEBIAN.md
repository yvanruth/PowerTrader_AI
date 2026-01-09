# PowerTrader AI - Debian/Proxmox VM Deployment Guide

## Overzicht

Ja, PowerTrader AI kan perfect draaien op een Proxmox VM met Debian 13.2.0. Deze guide legt uit hoe je het installeert en configureert.

## Vereisten

- Proxmox VM met Debian 13.2.0 (debian-13.2.0-amd64-netinst.iso)
- Minimaal 2GB RAM (4GB+ aanbevolen)
- Minimaal 10GB vrije schijfruimte
- Netwerkverbinding voor API calls (Binance/Kraken)
- GUI toegang nodig (VNC of X11 forwarding)

## Stap 1: Debian VM Installatie

1. **Installeer Debian 13.2.0** op je Proxmox VM
   - Kies "Graphical desktop environment" tijdens installatie (voor GUI)
   - Of installeer "Standard system utilities" en installeer later GUI packages

2. **Update het systeem:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

## Stap 2: Python & Dependencies Installeren

### Optie A: Met GUI (aanbevolen voor eerste setup)

```bash
# Installeer Python 3 en pip
sudo apt install -y python3 python3-pip python3-venv python3-tk

# Installeer GUI dependencies
sudo apt install -y xvfb x11vnc tigervnc-standalone-server
```

### Optie B: Headless (zonder GUI packages, alleen voor CLI)

```bash
# Installeer Python 3 en pip
sudo apt install -y python3 python3-pip python3-venv

# Voor headless GUI (Xvfb virtual display)
sudo apt install -y xvfb python3-tk
```

## Stap 3: Python Packages Installeren

```bash
# Clone of kopieer je repository naar de VM
cd ~
git clone https://github.com/yvanruth/PowerTrader_AI.git
cd PowerTrader_AI

# Installeer Python dependencies
pip3 install --user numpy matplotlib requests psutil cryptography numba
```

**Volledige lijst van dependencies:**
- `numpy` - Voor numerieke berekeningen
- `matplotlib` - Voor charts in de GUI
- `requests` - Voor API calls (Binance/Kraken)
- `psutil` - Voor process management
- `cryptography` - Voor Robinhood API (als je die gebruikt)
- `numba` - Optioneel, voor JIT compilation (performance)

## Stap 4: GUI Toegang Configureren

### Optie A: VNC Server (aanbevolen)

```bash
# Installeer VNC server
sudo apt install -y tigervnc-standalone-server

# Start VNC server (eerste keer vraagt om wachtwoord)
vncserver :1 -geometry 1920x1080 -depth 24

# Maak VNC auto-start (optioneel)
# Voeg toe aan ~/.bashrc of maak systemd service
```

**VNC verbinden:**
- Vanaf je lokale machine: gebruik VNC viewer
- Adres: `VM_IP_ADDRESS:5901` (of poort 5900 + display nummer)

### Optie B: X11 Forwarding (via SSH)

```bash
# Op je lokale machine (niet op de VM)
ssh -X gebruiker@VM_IP_ADDRESS

# Dan op de VM:
export DISPLAY=:10.0
python3 pt_hub.py
```

### Optie C: Xvfb (Headless met virtual display)

```bash
# Start virtual display
Xvfb :99 -screen 0 1920x1080x24 &

# Export DISPLAY variable
export DISPLAY=:99

# Start applicatie
python3 pt_hub.py
```

## Stap 5: Applicatie Configureren

1. **Kopieer API keys naar de VM:**
   ```bash
   # Plaats je API keys in de PowerTrader_AI folder
   nano kraken_key.txt
   nano kraken_secret.txt
   ```

2. **Configureer GUI settings:**
   ```bash
   # Start de GUI
   python3 pt_hub.py
   
   # Of headless:
   export DISPLAY=:99
   python3 pt_hub.py
   ```

## Stap 6: Auto-Start Configureren (Systemd Service)

Maak een systemd service om de applicatie automatisch te starten:

```bash
sudo nano /etc/systemd/system/powertrader.service
```

Voeg toe:
```ini
[Unit]
Description=PowerTrader AI Trading Bot
After=network.target

[Service]
Type=simple
User=jouw_gebruiker
WorkingDirectory=/home/jouw_gebruiker/PowerTrader_AI
Environment="DISPLAY=:99"
ExecStartPre=/usr/bin/Xvfb :99 -screen 0 1920x1080x24 &
ExecStart=/usr/bin/python3 /home/jouw_gebruiker/PowerTrader_AI/pt_hub.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Activeer en start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable powertrader
sudo systemctl start powertrader
```

## Stap 7: Firewall Configuratie

Als je VNC gebruikt, open poort 5901 (of je gekozen poort):

```bash
sudo ufw allow 5901/tcp
# Of voor iptables:
sudo iptables -A INPUT -p tcp --dport 5901 -j ACCEPT
```

## Troubleshooting

### "No display" error
```bash
# Zorg dat DISPLAY is gezet
export DISPLAY=:99
# Of voor VNC:
export DISPLAY=:1
```

### "Tkinter not found"
```bash
sudo apt install python3-tk
```

### "Permission denied" voor API keys
```bash
chmod 600 kraken_key.txt kraken_secret.txt
```

### GUI start niet
- Controleer of Xvfb draait: `ps aux | grep Xvfb`
- Controleer DISPLAY variable: `echo $DISPLAY`
- Test met: `xclock` (als geïnstalleerd)

## Performance Tips

1. **Geef de VM genoeg resources:**
   - Minimaal 2 CPU cores
   - 4GB+ RAM (training kan veel geheugen gebruiken)
   - SSD storage voor snellere I/O

2. **Monitor resources:**
   ```bash
   htop
   # Of
   watch -n 1 'free -h && df -h'
   ```

3. **Logs bekijken:**
   ```bash
   # Als systemd service:
   sudo journalctl -u powertrader -f
   ```

## Security Overwegingen

1. **Firewall:** Beperk toegang tot VNC poort (alleen vanaf vertrouwde IPs)
2. **SSH:** Gebruik key-based authentication
3. **API Keys:** Bewaar ze veilig, gebruik `chmod 600`
4. **Updates:** Houd Debian up-to-date: `sudo apt update && sudo apt upgrade`

## Backup Strategie

Belangrijke bestanden om te backuppen:
- `gui_settings.json`
- `kraken_key.txt` / `kraken_secret.txt`
- `hub_data/` folder
- Coin folders met training data (`BTC/`, `ETH/`, etc.)

```bash
# Maak een backup script
tar -czf powertrader_backup_$(date +%Y%m%d).tar.gz \
  gui_settings.json \
  kraken_*.txt \
  hub_data/ \
  BTC/ ETH/ XRP/ BNB/ DOGE/
```

## Conclusie

PowerTrader AI werkt prima op een Debian 13.2.0 Proxmox VM. De belangrijkste aandachtspunten zijn:
- GUI toegang configureren (VNC of Xvfb)
- Python dependencies installeren
- API keys veilig bewaren
- Systemd service voor auto-start (optioneel)

De applicatie kan 24/7 draaien op de VM en je kunt er via VNC of SSH+X11 forwarding bij.
