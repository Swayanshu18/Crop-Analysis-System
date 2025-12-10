import React, { useRef, useEffect } from 'react';

interface PestBox {
    box: number[]; // [x, y, w, h]
    label: string;
    confidence: number;
}

interface Props {
    imageUrl: string | null;
    boxes: PestBox[];
    maskBase64: string | null;
    maskShape: number[] | null; // [height, width]
}

const ResultsOverlay: React.FC<Props> = ({ imageUrl, boxes, maskBase64 }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!imageUrl || !canvasRef.current || !containerRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const img = new Image();
        img.src = imageUrl;
        img.onload = () => {
            // Set canvas size to match image
            canvas.width = img.width;
            canvas.height = img.height;

            // Draw image
            ctx.drawImage(img, 0, 0);

            // Draw Disease Mask if available
            if (maskBase64) {
                const maskImg = new Image();
                maskImg.src = `data:image/png;base64,${maskBase64}`;
                maskImg.onload = () => {
                    ctx.save();
                    ctx.globalCompositeOperation = "source-over";
                    ctx.globalAlpha = 0.4; // Transparency for overlay
                    ctx.drawImage(maskImg, 0, 0, img.width, img.height);
                    ctx.restore();

                    // Draw Boxes
                    drawBoxes(ctx, boxes);
                };
            } else {
                drawBoxes(ctx, boxes);
            }
        };
    }, [imageUrl, boxes, maskBase64]);

    const drawBoxes = (ctx: CanvasRenderingContext2D, boxes: PestBox[]) => {
        ctx.lineWidth = 3;
        ctx.font = "16px Arial";

        boxes.forEach((pest) => {
            const [x, y, w, h] = pest.box;

            // Draw Box
            ctx.strokeStyle = "#ef4444"; // Red
            ctx.strokeRect(x, y, w, h);

            // Draw Label Background
            ctx.fillStyle = "#ef4444";
            const text = `${pest.label} ${(pest.confidence * 100).toFixed(1)}%`;
            const textWidth = ctx.measureText(text).width;
            ctx.fillRect(x, y - 24, textWidth + 10, 24);

            // Draw Text
            ctx.fillStyle = "#ffffff";
            ctx.fillText(text, x + 5, y - 7);
        });
    };

    return (
        <div ref={containerRef} style={{
            position: 'relative',
            width: '100%',
            overflow: 'hidden',
            borderRadius: '0.5rem',
            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
            backgroundColor: '#fff',
            border: '1px solid #e2e8f0'
        }}>
            {imageUrl ? (
                <canvas ref={canvasRef} style={{ width: '100%', height: 'auto', display: 'block' }} />
            ) : (
                <div style={{
                    height: '16rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#9ca3af'
                }}>
                    No image uploaded
                </div>
            )}
        </div>
    );
};

export default ResultsOverlay;
