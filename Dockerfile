FROM python:3.13.3-slim

ARG USER_UID=760
RUN useradd --create-home --shell /bin/bash --uid ${USER_UID} evaluator

WORKDIR /home/evaluator/nicprint-clf

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY utils/ ./utils/
COPY models/ ./models/
COPY mappings/ ./mappings/
COPY .env ./

ENV PYTHONPATH=/home/evaluator/nicprint-clf/src
ENV MPLBACKEND=Agg
ENV MPLCONFIGDIR=/tmp/matplotlib

RUN mkdir -p figs results \
    && chown -R evaluator:evaluator /home/evaluator \
    && chmod 777 /home/evaluator

COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && chmod g+w /etc/passwd

USER evaluator

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]