# Relatório Técnico e Considerações Éticas: Contagem de Pedestres por Drone em Vigilância Urbana

**Disciplina:** Visão Computacional — Avaliação Técnica (Exercício 3, Item B)

---

## 1. Visão Geral da Arquitetura do Sistema
O sistema integra um detector de objetos em tempo real (**YOLOv4-tiny via OpenCV DNN**) com um rastreador por associação espacial (**IoU entre frames consecutivos, com predição linear de posição pela velocidade média; não há filtro de Kalman**) que atribui identificadores persistentes e preserva trilhas dos últimos 30 frames. Aplicado a aeronaves remotamente pilotadas (drones/RPAS) para monitoramento urbano, o pipeline realiza:
1. **Detecção frame a frame:** localização de pedestres e veículos com supressão de não-máximos (*NMS threshold 0.4*).
2. **Associação temporal:** IoU entre a caixa prevista de cada track e as detecções do frame atual.
3. **Contagem orientada a eventos:** entradas (track confirmado após 2 detecções) e saídas (track encerrado após 15 frames sem detecção), acumuladas.
4. **Métrica de estabilidade (estimativa heurística):** 6 prováveis ID switches em 1.32 min de vídeo (4.53/min), com 53 entradas e 44 saídas. A heurística conta apenas um objeto reidentificado sobre a posição de um track encerrado há poucos frames; trocas de ID entre dois objetos que se cruzam sem perder a detecção não são capturadas e não há gabarito (*ground truth*) para este vídeo, logo o valor tende a subestimar os ID switches reais. Custo do rastreador: 0.063 ms por frame.

---

## 2. Aplicação Técnica em Drones de Vigilância Urbana
O uso de drones equipados com este pipeline possibilita:
- **Gestão de grandes eventos e mobilidade:** mapeamento de densidade de multidões, identificação de gargalos de circulação em tempo real e apoio a rotas de evacuação em emergências.
- **Planejamento urbano:** estimativa de fluxo de pedestres em cruzamentos e praças para dimensionamento de faixas de travessia e calçadas.
- **Desafios específicos do ponto de vista aéreo (*bird's-eye view*):**
  - *Egomotion (movimento da câmera):* drones sofrem translação e guinada com o vento. Em voo pairado (*hovering*), a associação por IoU a taxas altas (>25 FPS) se mantém; com translação, é preciso compensar o movimento de fundo (fluxo óptico ou transformação afim entre frames) antes de associar.
  - *Escala reduzida:* pedestres vistos de altitudes elevadas têm menos de 30×30 pixels, exigindo sensores de alta resolução ou redes adaptadas a objetos pequenos; o YOLOv4-tiny a 416 px perde esses alvos.
  - *Restrição energética e térmica:* o processamento deve ocorrer a bordo com limite de potência (ordem de 5 a 15 W em NPUs como Jetson Orin Nano ou Hailo-8) para não comprometer a autonomia de voo (tipicamente 20 a 35 minutos).

---

## 3. Considerações Éticas e Conformidade Regulatória (LGPD / GDPR)

A vigilância urbana por plataformas aéreas suscita preocupações sérias de privacidade e liberdades civis:

### A. Privacidade por design (*privacy by design*) e minimização de dados
- **Processamento na borda (*edge computing*):** o vídeo bruto **não deve ser transmitido nem gravado** na íntegra em servidores centrais; inferência e rastreamento ocorrem a bordo.
- **Descarte imediato de frames:** após gerar os metadados agregados (por exemplo, "12 pessoas por minuto na zona sul"), os quadros são descartados da memória volátil.
- **Ausência de reconhecimento biométrico:** o pipeline opera com caixas delimitadoras de silhuetas, **sem reconhecimento facial** nem identificação civil.

### B. Riscos de vigilância em massa e efeito inibidor (*chilling effect*)
- A presença visível de drones pode constranger o exercício legítimo de reunião pública pacífica e a liberdade de locomoção.
- O emprego deve ser vinculado a finalidades legítimas e proporcionais de segurança e mobilidade, auditadas por conselhos de cidadania e pela autoridade de proteção de dados.

### C. Auditoria de vieses e erros operacionais
- Modelos treinados em visadas horizontais apresentam mais falsos negativos em tomadas zenitais, sob baixa iluminação, oclusão por copas de árvores ou sombras longas.
- Decisões automatizadas baseadas em contagens (por exemplo, fechar acessos por superlotação aparente) exigem supervisão humana (*human-in-the-loop*), sobretudo porque a própria métrica de estabilidade acima é uma estimativa.

### D. Transparência e notificação pública
- Operações de monitoramento aéreo em áreas urbanas devem ser precedidas de sinalização visível e aviso público sobre finalidade, órgão responsável e canais de contato do encarregado de proteção de dados.

---

## 4. Conclusão
O sistema é adequado para contagem quantitativa agregada com baixo custo computacional, desde que suas limitações (ausência de reidentificação por aparência, métrica de ID switch estimada, sensibilidade a escala e egomotion) sejam declaradas. A conformidade ética depende de anonimização arquitetural: a visão computacional como instrumento de medição estatística urbana, nunca de vigilância e identificação individual.