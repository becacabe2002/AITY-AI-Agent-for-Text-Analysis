# AITY Built-in: Ollama + the energy-estimation shim (ADR-003 §7.4)
# Pin the tag you benchmarked against instead of `latest`.
FROM ollama/ollama:latest

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*

COPY shim/requirements.txt /opt/shim/requirements.txt
RUN python3 -m venv /opt/shim/venv \
 && /opt/shim/venv/bin/pip install --no-cache-dir -r /opt/shim/requirements.txt

COPY shim/ /opt/shim/app/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# The base image's ENTRYPOINT is `ollama` with CMD `serve`; replace both.
ENTRYPOINT ["/entrypoint.sh"]
CMD []
