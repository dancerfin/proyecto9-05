#!/bin/bash
# Tráfico normal mejorado con más variedad

SERVERS=("10.1.1.5" "10.1.1.7" "10.1.1.9" "10.1.1.11" "10.1.1.13")
PROTOCOLS=("http" "icmp" "udp" "tcp")

while true; do
    target=${SERVERS[$RANDOM % ${#SERVERS[@]}]}
    protocol=${PROTOCOLS[$RANDOM % ${#PROTOCOLS[@]}]}
    
    case $protocol in
        "http")
            # Variedad de métodos HTTP
            METHODS=("GET" "POST" "HEAD")
            method=${METHODS[$RANDOM % ${#METHODS[@]}]}
            curl -X $method -m 2 "http://$target/test.html" &> /dev/null
            ;;
        "icmp")
            ping -c $((1 + RANDOM % 3)) $target &> /dev/null
            ;;
        "udp")
            hping3 -2 -c $((1 + RANDOM % 5)) -d $((64 + RANDOM % 500)) -p $((1024 + RANDOM % 64511)) $target &> /dev/null &
            ;;
        "tcp")
            hping3 -S -c $((1 + RANDOM % 5)) -d $((64 + RANDOM % 500)) -p $((1024 + RANDOM % 64511)) $target &> /dev/null &
            ;;
    esac
    
    sleep $((1 + RANDOM % 10))
done