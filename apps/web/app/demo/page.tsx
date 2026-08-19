import { Button } from "@/components/ui/button";
import { ModeToggle } from "@/components/ui/mode-toggle";
import React from "react";

function page() {
  return (
    <div>
      <Button variant="outline">Hello!</Button>
      <ModeToggle></ModeToggle>
    </div>
  );
}

export default page;
