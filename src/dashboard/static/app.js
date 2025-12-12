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
        const activeList = document.getElementById('positions-list'); // FIXED ID
        activeList.innerHTML = '';
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

                item.innerHTML = `
                    <div class="item-left">
                        <span class="symbol">${p.symbol}</span>
                        <span class="type badge">${p.type}</span>
                        <span class="entry">Entry: $${p.entry}</span>
                    </div>
                    <div class="item-center">
                       ${sparklineHtml}
                    </div>
                    <div class="item-right">
                        <div class="pnl ${colorClass}">$${p.pnl.toFixed(2)}</div>
                        <div class="pct ${colorClass}">${p.pct.toFixed(2)}%</div>
                    </div>
                `;
                activeList.appendChild(item);
            });
        } else {
            activeList.innerHTML = '<div class="empty-state">No open positions</div>';
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
                        <span class="entry" style="font-family: monospace; font-size: 11px;">${timeStr}</span>
                    </div>
                    <div class="item-right">
                        <div class="pnl ${colorClass}" style="font-size: 14px;">${s.signal}</div>
                        <div class="pct" style="font-size: 10px;">Conf: ${(s.conf * 100).toFixed(0)}%</div>
                    </div>
                `;
                sigList.appendChild(item);
            });
        } else {
            sigList.innerHTML = '<div class="empty-state">No recent signals</div>';
        }

        // Update Market Watch
        const priceList = document.getElementById('prices-list');
        if (priceList) {
            priceList.innerHTML = '';
            if (data.prices && data.prices.length > 0) {
                priceList.style.display = 'flex';
                priceList.style.flexWrap = 'wrap';
                priceList.style.gap = '8px';
                priceList.style.padding = '8px';

                data.prices.forEach(p => {
                    const chip = document.createElement('div');
                    chip.className = 'chip';
                    chip.style.display = 'flex';
                    chip.style.flexDirection = 'column';
                    chip.style.alignItems = 'center';
                    chip.style.padding = '4px 8px';

                    chip.innerHTML = `
                        <span style="font-weight:bold; font-size:0.8em; margin-bottom:2px;">${p.symbol}</span>
                        <span style="font-family:monospace; font-size:0.9em;">$${p.price.toFixed(p.price < 1 ? 4 : 2)}</span>
                     `;
                    priceList.appendChild(chip);
                });
            } else {
                priceList.innerHTML = '<div class="empty-state">Waiting for prices...</div>';
            }
        }

    } catch (e) {
        console.error("Failed to fetch stats", e);
    }
}

// Poll every 2 seconds
setInterval(fetchStats, 2000);
fetchStats();
