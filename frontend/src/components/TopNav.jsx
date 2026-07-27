import { Link, useNavigate, useLocation } from "react-router-dom";
import { ArrowLeft, Home } from "lucide-react";

export default function TopNav() {
  const navigate = useNavigate();
  const location = useLocation();

  if (location.pathname === "/") return null;

  const buttonClass =
    "flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3.5 py-1.5 text-xs font-medium text-foreground backdrop-blur-md shadow-[0_2px_10px_rgba(0,0,0,0.3)] transition-colors hover:bg-white/10";

  return (
    <div className="mb-4 flex items-center gap-2">
      <button onClick={() => navigate(-1)} className={buttonClass}>
        <ArrowLeft size={13} /> 뒤로
      </button>
      <Link to="/" className={buttonClass}>
        <Home size={13} /> 메인으로
      </Link>
    </div>
  );
}
