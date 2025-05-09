from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.lib.packet import in_proto
from ryu.lib.packet import ipv4
from ryu.lib.packet import icmp
from ryu.lib.packet import tcp
from ryu.lib.packet import udp
from ryu.lib.packet import arp
from ryu.lib import hub
import csv
import time
import math
import statistics
from datetime import datetime
from ml import MachineLearningAlgo

# Configuración global
APP_TYPE = 0  # 0: datacollection, 1: ddos detection
PREVENTION = 0  # DDoS prevention
TEST_TYPE = 0   # 0: normal, 1: attack (solo para datacollection)
INTERVAL = 5    # Intervalo de monitoreo en segundos
MIN_ATTACK_DURATION = 15  # Segundos mínimos para considerar ataque real

# Estructuras globales para múltiples switches
gflows = {}
iteration = {}
old_ssip_len = {}
prev_flow_count = {}
flow_cookie = {}
BLOCKED_PORTS = {}
keystore = {}
attack_start_time = {}  # Para rastrear inicio de ataques

# Funciones auxiliares
def get_iteration(dpid):
    global iteration
    iteration.setdefault(dpid, 0)
    return iteration[dpid]

def set_iteration(dpid, count):
    global iteration
    iteration[dpid] = count

def get_old_ssip_len(dpid):
    global old_ssip_len
    old_ssip_len.setdefault(dpid, 0)
    return old_ssip_len[dpid]

def set_old_ssip_len(dpid, count):
    global old_ssip_len
    old_ssip_len[dpid] = count

def get_prev_flow_count(dpid):
    global prev_flow_count
    prev_flow_count.setdefault(dpid, 0)
    return prev_flow_count[dpid]

def set_prev_flow_count(dpid, count):
    global prev_flow_count
    prev_flow_count[dpid] = count

def get_flow_number(dpid):
    global flow_cookie
    flow_cookie.setdefault(dpid, 0)
    flow_cookie[dpid] += 1
    return flow_cookie[dpid]

def get_time():
    return datetime.now()

def calculate_value(key, val):
    key = str(key).replace(".", "_")
    if key in keystore:
        oldval = keystore[key]
        cval = (val - oldval) 
        keystore[key] = val
        return cval
    else:
        keystore[key] = val
        return 0

# Funciones para manejo de archivos CSV
def init_portcsv(dpid):
    fname = f"switch_{dpid}_data.csv"
    with open(fname, 'a', buffering=1) as f:
        writ = csv.writer(f, delimiter=',')
        header = ["time", "sfe", "ssip", "rfip", "sdfp", "sdfb", "type"]
        writ.writerow(header)

def init_flowcountcsv(dpid):
    fname = f"switch_{dpid}_flowcount.csv"
    with open(fname, 'a', buffering=1) as f:
        writ = csv.writer(f, delimiter=',')
        header = ["time", "flowcount"]
        writ.writerow(header)

def update_flowcountcsv(dpid, row):
    fname = f"switch_{dpid}_flowcount.csv"
    with open(fname, 'a', buffering=1) as f:
        writ = csv.writer(f, delimiter=',')
        writ.writerow(row)

def update_portcsv(dpid, row):
    fname = f"switch_{dpid}_data.csv"
    with open(fname, 'a', buffering=1) as f:
        row.append(str(TEST_TYPE))
        writ = csv.writer(f, delimiter=',')
        writ.writerow(row)

def update_resultcsv(row):
    with open("result.csv", 'a', buffering=1) as f:
        row.append(str(TEST_TYPE))
        writ = csv.writer(f, delimiter=',')
        writ.writerow(row)

class DDoSML(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DDoSML, self).__init__(*args, **kwargs)
        self.mac_to_port = {}        # Mapeo MAC -> puerto por switch
        self.arp_ip_to_port = {}     # Mapeo ARP IP -> puerto por switch
        self.datapaths = {}          # Datapaths activos por switch ID
        self.mitigation = 0          # Estado de mitigación
        self.mlobj = None            # Objeto de ML
        self.attack_status = {}      # Estado de ataque por switch
        
        if APP_TYPE == 1:
            self.mlobj = MachineLearningAlgo()
            self.logger.info("Modo de detección DDoS (ML) activado")
        else:
            self.logger.info("Modo de colección de datos activado")
        
        self.flow_thread = hub.spawn(self._flow_monitor)

    def _flow_monitor(self):
        hub.sleep(INTERVAL * 2)
        while True:
            for dp in self.datapaths.values():
                self.request_flow_metrics(dp)
            hub.sleep(INTERVAL)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id
        
        self.datapaths[dpid] = datapath
        self.mac_to_port.setdefault(dpid, {})
        self.arp_ip_to_port.setdefault(dpid, {})
        BLOCKED_PORTS.setdefault(dpid, [])
        self.attack_status.setdefault(dpid, {'active': False, 'start_time': 0})

        # Flujo por defecto (table-miss)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                        ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions, get_flow_number(dpid))

        # Flujo para ARP
        match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_ARP)
        self.add_flow(datapath, 10, match, actions, get_flow_number(dpid))

        # Inicializar archivos CSV para este switch
        init_portcsv(dpid)
        init_flowcountcsv(dpid)
        self.logger.info(f"Switch {dpid} conectado")

    def request_flow_metrics(self, datapath):
        ofp_parser = datapath.ofproto_parser
        req = ofp_parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    def _speed_of_flow_entries(self, dpid, flows):
        curr_flow_count = len(flows)
        sfe = curr_flow_count - get_prev_flow_count(dpid)
        set_prev_flow_count(dpid, curr_flow_count)
        return sfe

    def _speed_of_source_ip(self, dpid, flows):
        ssip = set()
        for flow in flows:
            if 'ipv4_src' in flow.match:
                ssip.add(flow.match['ipv4_src'])
        
        cur_ssip_len = len(ssip)
        ssip_result = cur_ssip_len - get_old_ssip_len(dpid)
        set_old_ssip_len(dpid, cur_ssip_len)
        return ssip_result

    def _ratio_of_flowpair(self, dpid, flows):
        flow_count = max(len(flows) - 1, 1)  # Excluir table-miss
        interactive_flows = set()
        
        for flow in flows:
            if 'ipv4_src' in flow.match and 'ipv4_dst' in flow.match:
                src_ip = flow.match['ipv4_src']
                dst_ip = flow.match['ipv4_dst']
                flow_pair = frozenset({src_ip, dst_ip})
                interactive_flows.add(flow_pair)
        
        iflow = len(interactive_flows) * 2  # Cada par cuenta como 2 flujos
        return float(iflow) / flow_count if flow_count > 0 else 1.0

    def _stddev_packets(self, dpid, flows):
        packet_counts = []
        byte_counts = []
        hdr = f"switch_{dpid}"
        
        for flow in flows:
            if 'ipv4_src' in flow.match and 'ipv4_dst' in flow.match:
                src_ip = flow.match['ipv4_src']
                dst_ip = flow.match['ipv4_dst']
                
                byte_key = f"{hdr}_{src_ip}_{dst_ip}.bytes_count"
                pkt_key = f"{hdr}_{src_ip}_{dst_ip}.packets_count"
                
                byte_diff = calculate_value(byte_key, flow.byte_count)
                pkt_diff = calculate_value(pkt_key, flow.packet_count)
                
                byte_counts.append(byte_diff)
                packet_counts.append(pkt_diff)
        
        try:
            stddev_pkt = statistics.stdev(packet_counts) if packet_counts else 0
            stddev_byte = statistics.stdev(byte_counts) if byte_counts else 0
            return stddev_pkt, stddev_byte
        except:
            return 0, 0

    def _is_real_attack(self, dpid):
        """Verifica si el ataque es real basado en duración"""
        if not self.attack_status[dpid]['active']:
            self.attack_status[dpid]['start_time'] = time.time()
            self.attack_status[dpid]['active'] = True
            return False
        
        duration = time.time() - self.attack_status[dpid]['start_time']
        return duration >= MIN_ATTACK_DURATION

    def _reset_attack_status(self, dpid):
        """Resetea el estado de ataque"""
        self.attack_status[dpid]['active'] = False
        self.attack_status[dpid]['start_time'] = 0

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        flows = ev.msg.body
        
        gflows.setdefault(dpid, [])
        gflows[dpid].extend(flows)

        if ev.msg.flags == 0:  # Fin de la lista de flujos
            sfe = self._speed_of_flow_entries(dpid, gflows[dpid])
            ssip = self._speed_of_source_ip(dpid, gflows[dpid])
            rfip = self._ratio_of_flowpair(dpid, gflows[dpid])
            sdfp, sdfb = self._stddev_packets(dpid, gflows[dpid])

            if APP_TYPE == 1 and get_iteration(dpid) == 1:
                self.logger.info(f"Switch {dpid} - sfe:{sfe} ssip:{ssip} rfip:{rfip} sdfp:{sdfp} sdfb:{sdfb}")
                result = self.mlobj.classify([sfe, ssip, rfip, sdfp, sdfb])
                
                if '1' in result:
                    if self._is_real_attack(dpid):
                        self.logger.warning(f"¡Ataque DDoS detectado en Switch {dpid}!")
                        self.mitigation = 1
                        if PREVENTION == 1:
                            self._activate_prevention(dpid)
                    else:
                        self.logger.info(f"Posible ataque en Switch {dpid} - En evaluación...")
                else:
                    self.logger.info(f"Tráfico normal en Switch {dpid}")
                    self._reset_attack_status(dpid)
                    if self.mitigation == 1:
                        self._deactivate_prevention(dpid)
                        self.mitigation = 0
            else:
                t = get_time().strftime("%m/%d/%Y, %H:%M:%S")
                update_portcsv(dpid, [t, str(sfe), str(ssip), str(rfip), str(sdfp), str(sdfb)])
                update_resultcsv([str(sfe), str(ssip), str(rfip), str(sdfp), str(sdfb)])

            gflows[dpid] = []
            set_iteration(dpid, 1)
            update_flowcountcsv(dpid, [get_time().strftime("%m/%d/%Y, %H:%M:%S"), str(get_prev_flow_count(dpid))])

    def _activate_prevention(self, dpid):
        """Activa medidas de prevención mejoradas en el switch afectado"""
        datapath = self.datapaths.get(dpid)
        if not datapath:
            return
            
        self.logger.info(f"Iniciando prevención en Switch {dpid}")
        
        # Bloquear puertos sospechosos de forma más inteligente
        suspicious_ports = self._identify_suspicious_ports(dpid)
        
        for port in suspicious_ports:
            if port not in BLOCKED_PORTS[dpid]:
                self.block_port(datapath, port)
                BLOCKED_PORTS[dpid].append(port)
                self.logger.info(f"Switch {dpid}: Puerto {port} bloqueado por tráfico sospechoso")

    def _deactivate_prevention(self, dpid):
        """Desactiva medidas de prevención y limpia bloqueos"""
        datapath = self.datapaths.get(dpid)
        if not datapath:
            return
            
        self.logger.info(f"Finalizando prevención en Switch {dpid}")
        
        # Limpiar flujos de bloqueo
        for port in BLOCKED_PORTS[dpid]:
            self._remove_block_flow(datapath, port)
        
        BLOCKED_PORTS[dpid] = []
        self.logger.info(f"Switch {dpid}: Todos los puertos desbloqueados")

    def _identify_suspicious_ports(self, dpid):
        """Identifica puertos sospechosos basado en tráfico anómalo"""
        suspicious_ports = set()
        
        # Buscar puertos con alta actividad reciente
        for port, ip_list in self.arp_ip_to_port.get(dpid, {}).items():
            if len(ip_list) > 5:  # Más de 5 IPs diferentes en el puerto
                suspicious_ports.add(port)
        
        return list(suspicious_ports)

    def _remove_block_flow(self, datapath, portnumber):
        """Elimina reglas de bloqueo para un puerto"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(in_port=portnumber)
        
        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
            match=match,
            priority=100
        )
        datapath.send_msg(mod)

    def add_flow(self, datapath, priority, match, actions, serial_no, buffer_id=None, idletime=0, hardtime=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(
                datapath=datapath, cookie=serial_no, buffer_id=buffer_id,
                idle_timeout=idletime, hard_timeout=hardtime,
                priority=priority, match=match, instructions=inst)
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath, cookie=serial_no, priority=priority,
                idle_timeout=idletime, hard_timeout=hardtime,
                match=match, instructions=inst)
                
        datapath.send_msg(mod)

    def block_port(self, datapath, portnumber):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(in_port=portnumber)
        flow_serial_no = get_flow_number(datapath.id)
        self.add_flow(datapath, 100, match, [], flow_serial_no, hardtime=300)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("paquete truncado: %s de %s bytes",
                            ev.msg.msg_len, ev.msg.total_len)

        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        dpid = datapath.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        
        if not eth:
            return

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src

        # Inicializar estructuras para este switch
        self.mac_to_port.setdefault(dpid, {})
        self.arp_ip_to_port.setdefault(dpid, {})
        self.arp_ip_to_port[dpid].setdefault(in_port, [])
        BLOCKED_PORTS.setdefault(dpid, [])

        # Aprender dirección MAC
        self.mac_to_port[dpid][src] = in_port

        # Determinar puerto de salida
        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)

        # Manejo especial para ARP
        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            arp_pkt = pkt.get_protocol(arp.arp)
            if arp_pkt and arp_pkt.src_ip not in self.arp_ip_to_port[dpid][in_port]:
                self.arp_ip_to_port[dpid][in_port].append(arp_pkt.src_ip)

        # Instalar flujo si no es FLOOD
        if out_port != ofproto.OFPP_FLOOD:
            if eth.ethertype == ether_types.ETH_TYPE_IP:
                ip_pkt = pkt.get_protocol(ipv4.ipv4)
                if ip_pkt:
                    # Verificación de mitigación DDoS
                    if self.mitigation and PREVENTION:
                        if (in_port not in BLOCKED_PORTS[dpid] and 
                            ip_pkt.src not in self.arp_ip_to_port[dpid].get(in_port, [])):
                            self.logger.warning(f"Posible ataque desde {ip_pkt.src} en puerto {in_port}")
                            self.block_port(datapath, in_port)
                            BLOCKED_PORTS[dpid].append(in_port)
                            return

                    # Crear flujo normal
                    match = parser.OFPMatch(
                        in_port=in_port,
                        eth_type=ether_types.ETH_TYPE_IP,
                        ipv4_src=ip_pkt.src,
                        ipv4_dst=ip_pkt.dst)
                    
                    actions = [parser.OFPActionOutput(out_port)]
                    flow_serial_no = get_flow_number(dpid)
                    
                    if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                        self.add_flow(datapath, 1, match, actions, flow_serial_no, 
                                     buffer_id=msg.buffer_id)
                        return
                    else:
                        self.add_flow(datapath, 1, match, actions, flow_serial_no)

        # Enviar paquete
        actions = [parser.OFPActionOutput(out_port)]
        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

if __name__ == '__main__':
    from ryu.cmd import manager
    manager.main()