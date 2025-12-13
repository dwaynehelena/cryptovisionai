async function fetchStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        // Update Header
        document.getElementById('last-updated').innerText = `Updated: ${data.last_updated.split(' ')[1] || 'Today'}`;

        // Update Summaries
        document.getElementById('equity').innerText = `$${data.equity.toFixed(2)}`;
        document.getElementById('pnl').innerText = `$${data.pnl.toFixed(2)}`;

        const pnlEl = document.getElementById('pnl');
        const roiEl = document.getElementById('roi');

        pnlEl.className = `value ${data.pnl >= 0 ? 'positive' : 'negative'}`;
        roiEl.innerText = `${data.roi >= 0 ? '+' : ''}${data.roi.toFixed(2)}% ROI`;
        roiEl.className = `sub-value ${data.roi >= 0 ? 'positive' : 'negative'}`;

        // Update Days to Moonshot
        const daysEl = document.getElementById('days-to-moonshot');
        if (daysEl) {
            if (data.days_to_moonshot === undefined) {
                daysEl.innerText = 'Calculating...';
                daysEl.style.fontSize = '1em';
            } else if (data.days_to_moonshot < 0) {
                daysEl.innerText = 'Trend Flat';
                daysEl.style.fontSize = '0.9em'; // Smaller text for status
                daysEl.style.color = 'var(--sys-on-surface-variant)';
            } else if (data.days_to_moonshot === 0) {
                daysEl.innerText = 'MOON!';
                daysEl.className = 'value positive';
            } else {
                daysEl.innerText = `${data.days_to_moonshot.toFixed(1)} Days`;
                daysEl.className = 'value';
            }
        }

        function createSparkline(data, width = 100, height = 30) {
            if (!data || data.length < 2) return '';

            const min = Math.min(...data);
            const max = Math.max(...data);
            const range = max - min;
            if (range === 0) return ''; // Flat line

            const stepX = width / (data.length - 1);

            let path = `M 0 ${height - ((data[0] - min) / range * height)}`;

            data.forEach((val, i) => {
                if (i === 0) return;
                const x = i * stepX;
                const y = height - ((val - min) / range * height);
                path += ` L ${x} ${y}`;
            });

            const isPos = data[data.length - 1] >= data[0];
            const color = isPos ? 'var(--sys-success)' : 'var(--sys-error)';

            return `<svg width="${width}" height="${height}" style="overflow:visible">
                <path d="${path}" stroke="${color}" stroke-width="2" fill="none" />
                <circle cx="${width}" cy="${height - ((data[data.length - 1] - min) / range * height)}" r="2" fill="${color}" />
            </svg>`;
        }

        // Update Positions
        const activePosList = document.getElementById('positions-list');
        activePosList.innerHTML = '';
        if (data.positions && data.positions.length > 0) {
            data.positions.forEach(p => {
                const item = document.createElement('div');
                item.className = 'list-item';

                const isPos = p.pnl >= 0;
                const colorClass = isPos ? 'positive' : 'negative';

                let sparklineHtml = '';
                if (p.sparkline) {
                    sparklineHtml = createSparkline(p.sparkline);
                }

                // Estimate Value
                const val = p.amt * p.entry;

                item.innerHTML = `
                    <div class="item-left">
                        <span class="symbol">${p.symbol}</span>
                        <div style="display:flex; gap:8px;">
                            <span class="type badge">${p.type}</span>
                            <span style="font-size:0.75em; opacity:0.8; margin-top:2px;">Val: $${val.toFixed(0)}</span>
                        </div>
                        <span class="entry">Entry: $${p.entry}</span>
                    </div>
                    <div class="item-center">
                       ${sparklineHtml}
                       <div style="text-align:center; font-size:0.7em; opacity:0.5; margin-top:2px;">Live Price Action</div>
                    </div>
                    <div class="item-right">
                        <div class="pnl ${colorClass}">$${p.pnl.toFixed(2)}</div>
                        <div class="pct ${colorClass}">${p.pct.toFixed(2)}%</div>
                         <div style="text-align:right; font-size:0.7em; opacity:0.7;">Unrealized P&L</div>
                    </div>
                `;
                activePosList.appendChild(item);
            });
        } else {
            activePosList.innerHTML = '<div class="empty-state">No open positions</div>';
        }

        // Update Model Config
        if (data.model_config) {
            document.getElementById('train-features').innerText = data.model_config.train_features || '--';
            document.getElementById('model-features').innerText = data.model_config.model_features || '--';
            document.getElementById('feature-set').innerText = data.model_config.feature_set || '--';
        }

        // Update Feature Tags
        const tagContainer = document.getElementById('feature-tags');
        if (data.feature_list && data.feature_list.length > 0) {
            tagContainer.innerHTML = '';
            data.feature_list.forEach(feat => {
                const span = document.createElement('span');
                span.className = 'chip';
                span.style.fontSize = '0.75em';
                span.style.padding = '2px 8px';
                span.innerText = feat;
                tagContainer.appendChild(span);
            });
        }


        // Update Top Performers
        const topList = document.getElementById('top-performers-list');
        topList.innerHTML = '';
        if (data.top_performers && data.top_performers.length > 0) {
            // Usually 10 items
            data.top_performers.forEach(p => {
                const item = document.createElement('div');
                item.className = 'list-item compact-item';

                const isPos = p.perf >= 0;
                const colorClass = isPos ? 'positive' : 'negative';

                const valStr = p.unit === 'M' ? `$${p.perf.toFixed(2)}M` : `${p.perf.toFixed(2)}%`;

                item.innerHTML = `
                    <div class="item-left">
                        <span style="font-weight:bold; margin-right:8px; opacity:0.5;">#${p.rank}</span>
                        <span class="symbol">${p.symbol}</span>
                    </div>
                    <div class="item-right">
                        <div class="pct ${colorClass}">${valStr}</div>
                    </div>
                `;
                topList.appendChild(item);
            });
        }

        // Update Strategy Config
        if (data.strategy_config) {
            const modeEl = document.getElementById('strat-mode');
            if (modeEl && data.strategy_config.mode) {
                modeEl.innerText = data.strategy_config.mode;
            }
            document.getElementById('strat-risk').innerText = `${data.strategy_config.risk}%`;
            document.getElementById('strat-sl').innerText = `${data.strategy_config.sl}%`;
            document.getElementById('strat-tp').innerText = `${data.strategy_config.tp}%`;
            document.getElementById('strat-max').innerText = data.strategy_config.maxpos;
        }

        // Update Signals
        const sigList = document.getElementById('signals-list');
        sigList.innerHTML = '';
        if (data.signals && data.signals.length > 0) {
            // Limit to 5 signals to fit without scrolling
            data.signals.slice(0, 5).forEach(s => {
                const item = document.createElement('div');
                item.className = 'list-item compact-item';

                let colorClass = '';
                if (s.signal === 'BUY') colorClass = 'positive';
                else if (s.signal === 'SELL') colorClass = 'negative';

                // Parse Time: "2025-12-12 07:01:19,282" -> "07:01:19"
                let timeStr = s.time;
                try {
                    timeStr = s.time.split(',')[0].split(' ')[1];
                } catch (e) { }

                item.innerHTML = `
                    <div class="item-left">
                        <span class="symbol">${s.symbol}</span>
                        <div style="font-size:0.7em; opacity:0.7;">Time: ${timeStr}</div>
                        <div style="font-size:0.7em; opacity:0.7;">Conf: ${(s.conf * 100).toFixed(0)}%</div>
                    </div>
                    <div class="item-right">
                        <div class="pnl ${colorClass}" style="font-size: 14px;">${s.signal}</div>
                        <div class="pct" style="font-size: 10px;">Prob: High</div>
                    </div>
                `;
                sigList.appendChild(item);
            });
        } else {
            sigList.innerHTML = '<div class="empty-state">No recent signals</div>';
        }

        // Update Market Watch with Details (Prices) is fine.



        // Update Market Watch
        const priceList = document.getElementById('prices-list');
        if (priceList) {
            priceList.innerHTML = '';
            if (data.prices && data.prices.length > 0) {
                data.prices.forEach(p => {
                    const chip = document.createElement('div');
                    chip.className = 'chip';
                    chip.innerHTML = `<span style="font-weight:bold">${p.symbol}</span> <span>$${p.price.toFixed(p.price < 1 ? 4 : 2)}</span>`;
                    priceList.appendChild(chip);
                });
            } else {
                priceList.innerHTML = '<div style="opacity:0.5; font-size:0.8em">Waiting for prices...</div>';
            }
        }

        // Update Live Logs
        const logTerm = document.getElementById('log-terminal');
        if (logTerm && data.recent_logs && data.recent_logs.length > 0) {
            logTerm.innerHTML = '';
            data.recent_logs.forEach(line => {
                const lineDiv = document.createElement('div');
                lineDiv.innerText = line;
                if (line.includes("ERROR")) lineDiv.style.color = "#ff5555";
                else if (line.includes("WARNING")) lineDiv.style.color = "#ffaa00";
                else if (line.includes("INFO")) lineDiv.style.color = "#00ff00";
                logTerm.appendChild(lineDiv);
            });
            // Auto scroll parent
            logTerm.parentElement.scrollTop = logTerm.parentElement.scrollHeight;
        }

        // Update System Status
        if (data.system_status) {
            document.getElementById('log-size').innerText = data.system_status.log_size_mb;
            document.getElementById('disk-free').innerText = data.system_status.disk_free_gb;
        }

    } catch (e) {
        console.error("Failed to fetch stats", e);
    }
}

// Poll every 2 seconds
setInterval(fetchStats, 2000);
fetchStats();
